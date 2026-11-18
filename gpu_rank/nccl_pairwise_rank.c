/*
 * nccl_pairwise_rank.c -- rank every GPU in an NVL72 rack by NCCL
 * point-to-point bandwidth, from ONE mpirun launch.
 *
 * Why: in All-Reduce / All-to-All every GPU waits for the slowest one, so
 * nccl-tests reports one rack-wide busbw and a slow GPU stays hidden. Here
 * the job steps through N-1 rounds; in round k every rank r sends to
 * (r+k)%N and receives from (r-k+N)%N at the same time (all GPUs busy,
 * every GPU paired with a different partner each round). Each rank times
 * its own round. After N-1 rounds every ordered GPU pair has been measured
 * once. A slow GPU drags down all 2(N-1) measurements it takes part in,
 * while a healthy GPU meets it in only 2 of its rounds -- so each GPU's
 * average over its rounds sorts the rack cleanly, slowest first.
 *
 * Same NCCL path as the rack test (P2P/MNNVL over NVLink): it is launched
 * with the same mpirun flags and NCCL env as run_nccl_test.sh -- see
 * run_gpu_rank.sh.
 *
 * Build-friendly on purpose: NCCL and the CUDA runtime are loaded with
 * dlopen() at run time and MPI is not linked at all (rank/size/local rank
 * come from Open MPI's env vars, the NCCL bootstrap from NCCL_COMM_ID). So
 * it needs NO CUDA/NCCL/MPI headers -- any aarch64 gcc builds it:
 *     gcc -O2 -std=gnu11 -o nccl_pairwise_rank nccl_pairwise_rank.c -ldl
 *
 *   nccl_pairwise_rank [-b 1G] [-n 10] [-w 3] [-o matrix.csv]
 *   nccl_pairwise_rank --selftest [N] [slow_rank] [factor]   (no GPU needed)
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ---- minimal NCCL / CUDA runtime ABI (stable across NCCL 2.x, CUDA 12/13) */
typedef struct { char internal[128]; } ncclUniqueId;
typedef void *ncclComm_t;
typedef void *cudaStream_t;
typedef int ncclResult_t;
typedef int cudaError_t;
enum { ncclInt8 = 0 };          /* everything is moved as raw bytes */
enum { ncclSum = 0 };
enum { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 };

static struct {
  ncclResult_t (*GetUniqueId)(ncclUniqueId *);
  ncclResult_t (*CommInitRank)(ncclComm_t *, int, ncclUniqueId, int);
  ncclResult_t (*CommDestroy)(ncclComm_t);
  ncclResult_t (*GroupStart)(void);
  ncclResult_t (*GroupEnd)(void);
  ncclResult_t (*Send)(const void *, size_t, int, int, ncclComm_t, cudaStream_t);
  ncclResult_t (*Recv)(void *, size_t, int, int, ncclComm_t, cudaStream_t);
  ncclResult_t (*AllGather)(const void *, void *, size_t, int, ncclComm_t, cudaStream_t);
  ncclResult_t (*AllReduce)(const void *, void *, size_t, int, int, ncclComm_t, cudaStream_t);
  const char *(*GetErrorString)(ncclResult_t);
} nccl;

static struct {
  cudaError_t (*SetDevice)(int);
  cudaError_t (*Malloc)(void **, size_t);
  cudaError_t (*Memset)(void *, int, size_t);
  cudaError_t (*Memcpy)(void *, const void *, size_t, int);
  cudaError_t (*StreamCreate)(cudaStream_t *);
  cudaError_t (*StreamSynchronize)(cudaStream_t);
  cudaError_t (*DeviceGetPCIBusId)(char *, int, int);
  const char *(*GetErrorString)(cudaError_t);
} cuda;

static int g_rank = 0;

#define DIE(...) do { fprintf(stderr, "[rank %d] ERROR: ", g_rank); fprintf(stderr, __VA_ARGS__); \
                      fprintf(stderr, "\n"); exit(1); } while (0)
#define NCCLCHECK(x) do { ncclResult_t r_ = (x); if (r_ != 0) DIE("%s failed: %s", #x, nccl.GetErrorString(r_)); } while (0)
#define CUDACHECK(x) do { cudaError_t e_ = (x); if (e_ != 0) DIE("%s failed: %s", #x, cuda.GetErrorString(e_)); } while (0)

static void *open_first(const char *const *names) {
  for (; *names; names++) {
    void *h = dlopen(*names, RTLD_NOW | RTLD_GLOBAL);
    if (h) return h;
  }
  return NULL;
}

static void *sym(void *h, const char *name) {
  void *p = dlsym(h, name);
  if (!p) DIE("symbol %s not found (%s)", name, dlerror());
  return p;
}

static void load_libs(void) {
  static const char *const nccl_names[] = {"libnccl.so.2", "libnccl.so", NULL};
  static const char *const cudart_names[] = {"libcudart.so.13", "libcudart.so.12", "libcudart.so", NULL};
  void *hn = open_first(nccl_names);
  if (!hn) DIE("cannot dlopen libnccl.so.2 -- is the pack's lib/ in LD_LIBRARY_PATH?");
  void *hc = open_first(cudart_names);
  if (!hc) DIE("cannot dlopen libcudart.so.{13,12} -- is the pack's lib/ or /usr/local/cuda/lib64 in LD_LIBRARY_PATH?");
  *(void **)&nccl.GetUniqueId = sym(hn, "ncclGetUniqueId");
  *(void **)&nccl.CommInitRank = sym(hn, "ncclCommInitRank");
  *(void **)&nccl.CommDestroy = sym(hn, "ncclCommDestroy");
  *(void **)&nccl.GroupStart = sym(hn, "ncclGroupStart");
  *(void **)&nccl.GroupEnd = sym(hn, "ncclGroupEnd");
  *(void **)&nccl.Send = sym(hn, "ncclSend");
  *(void **)&nccl.Recv = sym(hn, "ncclRecv");
  *(void **)&nccl.AllGather = sym(hn, "ncclAllGather");
  *(void **)&nccl.AllReduce = sym(hn, "ncclAllReduce");
  *(void **)&nccl.GetErrorString = sym(hn, "ncclGetErrorString");
  *(void **)&cuda.SetDevice = sym(hc, "cudaSetDevice");
  *(void **)&cuda.Malloc = sym(hc, "cudaMalloc");
  *(void **)&cuda.Memset = sym(hc, "cudaMemset");
  *(void **)&cuda.Memcpy = sym(hc, "cudaMemcpy");
  *(void **)&cuda.StreamCreate = sym(hc, "cudaStreamCreate");
  *(void **)&cuda.StreamSynchronize = sym(hc, "cudaStreamSynchronize");
  *(void **)&cuda.DeviceGetPCIBusId = sym(hc, "cudaDeviceGetPCIBusId");
  *(void **)&cuda.GetErrorString = sym(hc, "cudaGetErrorString");
}

static int env_int(const char *a, const char *b) {
  const char *v = getenv(a);
  if (!v && b) v = getenv(b);
  if (!v) DIE("%s not set -- launch with Open MPI's mpirun (see run_gpu_rank.sh)", a);
  return atoi(v);
}

static size_t parse_size(const char *s) {
  char *end;
  double v = strtod(s, &end);
  switch (*end) {
    case 'K': case 'k': v *= 1024.0; break;
    case 'M': case 'm': v *= 1024.0 * 1024; break;
    case 'G': case 'g': v *= 1024.0 * 1024 * 1024; break;
  }
  return (size_t)v;
}

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- analysis (rank 0 only; also driven by --selftest) ------------------
 * B[i*N + k] = GB/s rank i measured in round k (k = 1..N-1; column 0 unused).
 * In round k, i sent to (i+k)%N and received from (i-k+N)%N, so each
 * measurement involves three GPUs: i, its send peer and its recv peer.
 * score(g)  = mean of every measurement g was involved in (3(N-1) values).
 * own(g)    = mean of g's own N-1 measurements.
 * worst(g)  = the other GPU whose shared measurements with g are lowest --
 *             for every healthy GPU that converges on the slow one.     */
#define HOSTLEN 64
#define BUSLEN 32

static int cmp_dbl(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static double median(const double *v, int n) {
  double *t = malloc(n * sizeof *t);
  memcpy(t, v, n * sizeof *t);
  qsort(t, n, sizeof *t, cmp_dbl);
  double m = (n % 2) ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
  free(t);
  return m;
}

static const double *g_score;
static int cmp_by_score(const void *a, const void *b) {
  double x = g_score[*(const int *)a], y = g_score[*(const int *)b];
  return (x > y) - (x < y);
}

static void analyze(int N, const double *B, const char *hosts, const char *buses,
                    size_t bytes, int iters, const char *csv) {
  double *score = calloc(N, sizeof *score), *own = calloc(N, sizeof *own);
  double *cnt = calloc(N, sizeof *cnt);
  double *psum = calloc((size_t)N * N, sizeof *psum), *pcnt = calloc((size_t)N * N, sizeof *pcnt);
  double *worstv = calloc(N, sizeof *worstv);
  int *worstp = calloc(N, sizeof *worstp), *order = malloc(N * sizeof *order);
  for (int i = 0; i < N; i++) {
    for (int k = 1; k < N; k++) {
      double v = B[(size_t)i * N + k];
      int set[3] = {i, (i + k) % N, (i - k + N) % N};
      int n = set[1] == set[2] ? 2 : 3;              /* k == N/2: same peer both ways */
      own[i] += v / (N - 1);
      for (int a = 0; a < n; a++) {
        score[set[a]] += v; cnt[set[a]] += 1;
        for (int b = 0; b < n; b++)
          if (a != b) { psum[(size_t)set[a] * N + set[b]] += v; pcnt[(size_t)set[a] * N + set[b]] += 1; }
      }
    }
  }
  for (int g = 0; g < N; g++) {
    score[g] /= cnt[g];
    worstv[g] = 1e30; worstp[g] = -1;
    for (int x = 0; x < N; x++) {
      double c = pcnt[(size_t)g * N + x];
      if (x == g || c == 0) continue;
      double m = psum[(size_t)g * N + x] / c;
      if (m < worstv[g]) { worstv[g] = m; worstp[g] = x; }
    }
    order[g] = g;
  }
#define M(i, j) B[(size_t)(i) * N + (((j) - (i) + N) % N)]
  g_score = score;
  qsort(order, N, sizeof *order, cmp_by_score);
  double med = median(score, N);

  printf("\n=== GPU ranking: slowest -> fastest (%d GPUs, %.0f MiB per transfer, %d iters/round) ===\n",
         N, bytes / 1048576.0, iters);
  printf("score = mean NCCL send/recv GB/s (one direction) of every measurement the GPU took part in\n");
  printf("rack median score: %.1f GB/s\n\n", med);
  printf("%4s %5s  %-20s %-14s %9s %6s %9s  %s\n", "#", "rank", "host", "busId", "score", "vsMed", "own", "worst partner (shared GB/s)");
  for (int p = 0; p < N; p++) {
    int g = order[p];
    printf("%4d %5d  %-20s %-14s %9.1f %5.1f%% %9.1f  <-> rank %d %s %s (%.1f)\n", p + 1, g,
           hosts + (size_t)g * HOSTLEN, buses + (size_t)g * BUSLEN, score[g], 100.0 * score[g] / med, own[g],
           worstp[g], hosts + (size_t)worstp[g] * HOSTLEN, buses + (size_t)worstp[g] * BUSLEN, worstv[g]);
  }
  printf("\nRead it top-down: a GPU clearly below the rack median (e.g. < 95%%) whose 'own' column is\n"
         "also low is the suspect; healthy GPUs list that same suspect as their worst partner.\n");

  if (csv) {
    FILE *f = fopen(csv, "w");
    if (!f) { perror(csv); }
    else {
      fprintf(f, "from\\to");
      for (int j = 0; j < N; j++) fprintf(f, ",%d", j);
      fprintf(f, "\n");
      for (int i = 0; i < N; i++) {
        fprintf(f, "%d", i);
        for (int j = 0; j < N; j++) i == j ? fprintf(f, ",") : fprintf(f, ",%.2f", M(i, j));
        fprintf(f, "\n");
      }
      fclose(f);
      printf("pair matrix (GB/s, row = sender's measurement): %s\n", csv);
    }
  }
#undef M
  free(score); free(own); free(cnt); free(psum); free(pcnt); free(worstv); free(worstp); free(order);
}

static int selftest(int argc, char **argv) {
  int N = argc > 2 ? atoi(argv[2]) : 72, slow = argc > 3 ? atoi(argv[3]) : 37;
  double factor = argc > 4 ? atof(argv[4]) : 0.7;
  double *B = calloc((size_t)N * N, sizeof *B);
  char *hosts = calloc((size_t)N, HOSTLEN), *buses = calloc((size_t)N, BUSLEN);
  srand(1);
  for (int i = 0; i < N; i++) {
    snprintf(hosts + (size_t)i * HOSTLEN, HOSTLEN, "node%02d", i / 4);
    snprintf(buses + (size_t)i * BUSLEN, BUSLEN, "0000:%02x:00.0", 0x8 + i % 4);
    for (int k = 1; k < N; k++) {
      int to = (i + k) % N, from = (i - k + N) % N;
      double v = 700.0 * (1.0 + ((rand() % 100) - 50) / 5000.0);   /* +-1% noise */
      if (i == slow || to == slow || from == slow) v *= factor;
      B[(size_t)i * N + k] = v;
    }
  }
  printf("selftest: %d GPUs, rank %d at %.0f%% link speed\n", N, slow, factor * 100);
  analyze(N, B, hosts, buses, (size_t)1 << 30, 10, NULL);
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 1 && !strcmp(argv[1], "--selftest")) return selftest(argc, argv);

  size_t bytes = (size_t)1 << 30;
  int iters = 10, warmup = 3;
  const char *csv = NULL;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-b") && i + 1 < argc) bytes = parse_size(argv[++i]);
    else if (!strcmp(argv[i], "-n") && i + 1 < argc) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-w") && i + 1 < argc) warmup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-o") && i + 1 < argc) csv = argv[++i];
    else { fprintf(stderr, "usage: %s [-b 1G] [-n 10] [-w 3] [-o matrix.csv] | --selftest [N] [slow] [factor]\n", argv[0]); return 1; }
  }

  g_rank = env_int("OMPI_COMM_WORLD_RANK", "PMIX_RANK");
  int N = env_int("OMPI_COMM_WORLD_SIZE", NULL);
  int local = env_int("OMPI_COMM_WORLD_LOCAL_RANK", NULL);
  if (!getenv("NCCL_COMM_ID")) DIE("NCCL_COMM_ID not set (ip:port of rank 0's host) -- use run_gpu_rank.sh");
  if (N < 2) DIE("need at least 2 ranks");

  load_libs();
  CUDACHECK(cuda.SetDevice(local));
  cudaStream_t s;
  CUDACHECK(cuda.StreamCreate(&s));

  /* With NCCL_COMM_ID set, every rank derives the same id and rank 0 hosts
   * the bootstrap root at that address -- no MPI needed. */
  ncclUniqueId id;
  ncclComm_t comm;
  NCCLCHECK(nccl.GetUniqueId(&id));
  NCCLCHECK(nccl.CommInitRank(&comm, N, id, g_rank));

  void *sbuf, *rbuf, *tiny;
  CUDACHECK(cuda.Malloc(&sbuf, bytes));
  CUDACHECK(cuda.Malloc(&rbuf, bytes));
  CUDACHECK(cuda.Malloc(&tiny, 4096 + (size_t)N * N * sizeof(double) + (size_t)N * (HOSTLEN + BUSLEN) * 2));
  CUDACHECK(cuda.Memset(sbuf, 1, bytes));

  double *row = calloc(N, sizeof *row);
  double t_start = now_s();
  if (g_rank == 0) printf("nccl_pairwise_rank: %d ranks, %d rounds x (%d warmup + %d timed) x %.0f MiB\n",
                          N, N - 1, warmup, iters, bytes / 1048576.0), fflush(stdout);

  for (int k = 1; k < N; k++) {
    int to = (g_rank + k) % N, from = (g_rank - k + N) % N;
    /* barrier: a tiny all-reduce, so no rank starts round k while its
     * partners are still finishing round k-1 */
    NCCLCHECK(nccl.AllReduce(tiny, tiny, 1, ncclInt8, ncclSum, comm, s));
    CUDACHECK(cuda.StreamSynchronize(s));
    double t0 = 0;
    for (int it = 0; it < warmup + iters; it++) {
      if (it == warmup) { CUDACHECK(cuda.StreamSynchronize(s)); t0 = now_s(); }
      NCCLCHECK(nccl.GroupStart());
      NCCLCHECK(nccl.Send(sbuf, bytes, ncclInt8, to, comm, s));
      NCCLCHECK(nccl.Recv(rbuf, bytes, ncclInt8, from, comm, s));
      NCCLCHECK(nccl.GroupEnd());
    }
    CUDACHECK(cuda.StreamSynchronize(s));
    row[k] = (double)bytes * iters / (now_s() - t0) / 1e9;
    if (g_rank == 0 && (k % 10 == 0 || k == N - 1))
      printf("  round %d/%d done (%.0fs)\n", k, N - 1, now_s() - t_start), fflush(stdout);
  }

  /* Gather every row + host/busId to all ranks via NCCL (rank 0 prints). */
  char host[HOSTLEN] = {0}, bus[BUSLEN] = {0};
  gethostname(host, HOSTLEN - 1);
  CUDACHECK(cuda.DeviceGetPCIBusId(bus, BUSLEN, local));
  char *dev = tiny;
  char *d_row = dev + 4096, *d_all = d_row + N * sizeof(double);
  char *d_id = d_all + (size_t)N * N * sizeof(double), *d_ids = d_id + HOSTLEN + BUSLEN;
  char idbuf[HOSTLEN + BUSLEN];
  memcpy(idbuf, host, HOSTLEN);
  memcpy(idbuf + HOSTLEN, bus, BUSLEN);
  CUDACHECK(cuda.Memcpy(d_row, row, N * sizeof(double), cudaMemcpyHostToDevice));
  CUDACHECK(cuda.Memcpy(d_id, idbuf, sizeof idbuf, cudaMemcpyHostToDevice));
  NCCLCHECK(nccl.GroupStart());
  NCCLCHECK(nccl.AllGather(d_row, d_all, N * sizeof(double), ncclInt8, comm, s));
  NCCLCHECK(nccl.AllGather(d_id, d_ids, sizeof idbuf, ncclInt8, comm, s));
  NCCLCHECK(nccl.GroupEnd());
  CUDACHECK(cuda.StreamSynchronize(s));

  if (g_rank == 0) {
    double *B = malloc((size_t)N * N * sizeof *B);
    char *ids = malloc((size_t)N * sizeof idbuf);
    char *hosts = malloc((size_t)N * HOSTLEN), *buses = malloc((size_t)N * BUSLEN);
    CUDACHECK(cuda.Memcpy(B, d_all, (size_t)N * N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDACHECK(cuda.Memcpy(ids, d_ids, (size_t)N * sizeof idbuf, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
      memcpy(hosts + (size_t)i * HOSTLEN, ids + (size_t)i * sizeof idbuf, HOSTLEN);
      memcpy(buses + (size_t)i * BUSLEN, ids + (size_t)i * sizeof idbuf + HOSTLEN, BUSLEN);
    }
    printf("measurement done in %.0fs\n", now_s() - t_start);
    analyze(N, B, hosts, buses, bytes, iters, csv);
    fflush(stdout);
  }
  NCCLCHECK(nccl.CommDestroy(comm));
  return 0;
}
