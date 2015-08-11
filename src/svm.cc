/*
 This file originates from LibSVM <http://www.csie.ntu.edu.tw/~cjlin/libsvm/>

 Copyright (c) 2000-2013 Chih-Chung Chang and Chih-Jen Lin
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:

 1. Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.

 3. Neither name of copyright holders nor the names of its contributors
 may be used to endorse or promote products derived from this software
 without specific prior written permission.


 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/* This file is part of the Princeton FCMA Toolbox-- namely
   the "svm_cross_validation_no_shuffle" function, an edited
   version of the original.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <iostream>
#include "common.h"
#include "svm.h"

int libsvm_version = LIBSVM_VERSION;
#ifndef min
template <class T>
static inline T min(T x, T y) {
  return (x < y) ? x : y;
}
#endif
#ifndef max
template <class T>
static inline T max(T x, T y) {
  return (x > y) ? x : y;
}
#endif
template <class T>
static inline void swap(T &x, T &y) {
  T t = x;
  x = y;
  y = t;
}
template <class S, class T>
static inline void clone(T *&dst, S *src, int n) {
  dst = new T[n];
  memcpy((void *)dst, (void *)src, sizeof(T) * n);
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type, n) (type *) malloc((n) * sizeof(type))
#ifndef Calloc
#define Calloc(type, n) (type *) calloc(n, sizeof(type))
#endif

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache {
 public:
  Cache(int l, long int size);
  ~Cache();

  // request data [0,len)
  // return some position p where [p,len) need to be filled
  // (p >= len if nothing needs to be filled)
  int get_data(const int index, Qfloat **data, int len);
  void swap_index(int i, int j);

 private:
  int l;
  long int size;
  struct head_t {
    head_t *prev, *next;  // a circular list
    Qfloat *data;
    int len;  // data[0,len) is cached in this entry
  };

  head_t *head;
  head_t lru_head;
  void lru_delete(head_t *h);
  void lru_insert(head_t *h);
};

Cache::Cache(int l_, long int size_) : l(l_), size(size_) {
  head = (head_t *)calloc(l, sizeof(head_t));  // initialized to 0
  size /= sizeof(Qfloat);
  size -= l * sizeof(head_t) / sizeof(Qfloat);
  size =
      max(size, 2 * (long int)l);  // cache must be large enough for two columns
  lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache() {
  for (head_t *h = lru_head.next; h != &lru_head; h = h->next) free(h->data);
  free(head);
}

void Cache::lru_delete(head_t *h) {
  // delete from current location
  h->prev->next = h->next;
  h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h) {
  // insert to last position
  h->next = &lru_head;
  h->prev = lru_head.prev;
  h->prev->next = h;
  h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len) {
  head_t *h = &head[index];
  if (h->len) lru_delete(h);
  int more = len - h->len;

  if (more > 0) {
    // free old space
    while (size < more) {
      head_t *old = lru_head.next;
      lru_delete(old);
      free(old->data);
      size += old->len;
      old->data = 0;
      old->len = 0;
    }

    // allocate new space
    h->data = (Qfloat *)realloc(h->data, sizeof(Qfloat) * len);
    size -= more;
    swap(h->len, len);
  }

  lru_insert(h);
  *data = h->data;
  return len;
}

void Cache::swap_index(int i, int j) {
  if (i == j) return;

  if (head[i].len) lru_delete(&head[i]);
  if (head[j].len) lru_delete(&head[j]);
  swap(head[i].data, head[j].data);
  swap(head[i].len, head[j].len);
  if (head[i].len) lru_insert(&head[i]);
  if (head[j].len) lru_insert(&head[j]);

  if (i > j) swap(i, j);
  for (head_t *h = lru_head.next; h != &lru_head; h = h->next) {
    if (h->len > i) {
      if (h->len > j)
        swap(h->data[i], h->data[j]);
      else {
        // give up
        lru_delete(h);
        free(h->data);
        size += h->len;
        h->data = 0;
        h->len = 0;
      }
    }
  }
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
 public:
  virtual Qfloat *get_Q(int column, int len) const = 0;
  virtual double *get_QD() const = 0;
  virtual void swap_index(int i, int j) const = 0;
  virtual ~QMatrix() {}
};

class Kernel : public QMatrix {
 public:
  Kernel(int l, svm_node *const *x, const svm_parameter &param);
  virtual ~Kernel();

  static double k_function(const svm_node *x, const svm_node *y,
                           const svm_parameter &param, int l);
  virtual Qfloat *get_Q(int column, int len) const = 0;
  virtual double *get_QD() const = 0;
  virtual void swap_index(int i, int j) const  // no so const...
  {
    swap(x[i], x[j]);
  }
  const svm_node **x;

 protected:
  double (Kernel::*kernel_function)(int i, int j) const;

 private:
  int length;

  // svm_parameter
  const int kernel_type;
  const int degree;
  const double gamma;
  const double coef0;

  static double dot(const svm_node *px, const svm_node *py, int l);
  double kernel_linear(int i, int j) const { return dot(x[i], x[j], length); }
  double kernel_precomputed(int i, int j) const {
    return x[i][(int)(x[j][0].value)].value;
  }
};

// need to figure out the data layout of x_!!!!!
Kernel::Kernel(int l, svm_node *const *x_, const svm_parameter &param)
    : kernel_type(param.kernel_type),
      degree(param.degree),
      gamma(param.gamma),
      coef0(param.coef0) {
  switch (kernel_type) {
    case LINEAR:
      kernel_function = &Kernel::kernel_linear;
      break;
    case PRECOMPUTED:
      kernel_function = &Kernel::kernel_precomputed;
      break;
  }

  clone(x, x_, l);

  svm_node *temp_x = x_[0];
  length = 0;
  while (temp_x->index != -1) {
    length++;
    temp_x++;
  }
}

Kernel::~Kernel() { delete[] x; }

double Kernel::dot(const svm_node *px, const svm_node *py, int length) {
  // we assume px and py always have the same length and there's no index
  // missing
  float sum = cblas_sdot(length, (float *)px + 1, 2, (float *)py + 1, 2);
  return sum;
}

double Kernel::k_function(const svm_node *x, const svm_node *y,
                          const svm_parameter &param, int l) {
  switch (param.kernel_type) {
    case LINEAR:
      return dot(x, y, l);
    case PRECOMPUTED:  // x: test (validation), y: SV
      return x[(int)(y->value)].value;
    default:
      return 0;  // Unreachable
  }
}

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver {
 public:
  Solver() {};
  virtual ~Solver() {};

  struct SolutionInfo {
    double obj;
    double rho;
    double upper_bound_p;
    double upper_bound_n;
    double r;  // for Solver_NU
  };

  void Solve(int l, const QMatrix &Q, const double *p_, const schar *y_,
             double *alpha_, double Cp, double Cn, double eps, SolutionInfo *si,
             int shrinking);

 protected:
  int active_size;
  schar *y;
  double *G;  // gradient of objective function
  enum {
    LOWER_BOUND,
    UPPER_BOUND,
    FREE
  };
  char *alpha_status;  // LOWER_BOUND, UPPER_BOUND, FREE
  double *alpha;
  const QMatrix *Q;
  const double *QD;
  double eps;
  double Cp, Cn;
  double *p;
  int *active_set;
  double *G_bar;  // gradient, if we treat free variables as 0
  int l;
  bool unshrink;  // XXX

  double get_C(int i) { return (y[i] > 0) ? Cp : Cn; }
  void update_alpha_status(int i) {
    if (alpha[i] >= get_C(i))
      alpha_status[i] = UPPER_BOUND;
    else if (alpha[i] <= 0)
      alpha_status[i] = LOWER_BOUND;
    else
      alpha_status[i] = FREE;
  }
  bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
  bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
  bool is_free(int i) { return alpha_status[i] == FREE; }
  void swap_index(int i, int j);
  void reconstruct_gradient();
  virtual int select_working_set(int &i, int &j);
  virtual double calculate_rho();
  virtual void do_shrinking();

 private:
  bool be_shrunk(int i, double Gmax1, double Gmax2);
};

void Solver::swap_index(int i, int j) {
  Q->swap_index(i, j);
  swap(y[i], y[j]);
  swap(G[i], G[j]);
  swap(alpha_status[i], alpha_status[j]);
  swap(alpha[i], alpha[j]);
  swap(p[i], p[j]);
  swap(active_set[i], active_set[j]);
  swap(G_bar[i], G_bar[j]);
}

void Solver::reconstruct_gradient() {
  // reconstruct inactive elements of G from G_bar and free variables

  if (active_size == l) return;

  int i, j;
  int nr_free = 0;

  for (j = active_size; j < l; j++) G[j] = G_bar[j] + p[j];

#pragma simd reduction(+ : nr_free)
  for (j = 0; j < active_size; j++) nr_free += alpha_status[j] == FREE ? 1 : 0;

  if (nr_free * l > 2 * active_size * (l - active_size)) {
    for (i = active_size; i < l; i++) {
      const Qfloat *Q_i = Q->get_Q(i, active_size);
      for (j = 0; j < active_size; j++)
        G[i] += alpha_status[j] == FREE ? alpha[j] * Q_i[j] : 0;
    }
  } else {
    for (i = 0; i < active_size; i++)
      if (is_free(i)) {
        const Qfloat *Q_i = Q->get_Q(i, l);
        double alpha_i = alpha[i];
        for (j = active_size; j < l; j++) G[j] += alpha_i * Q_i[j];
      }
  }
}

void Solver::Solve(int l, const QMatrix &Q, const double *p_, const schar *y_,
                   double *alpha_, double Cp, double Cn, double eps,
                   SolutionInfo *si, int shrinking) {
  this->l = l;
  this->Q = &Q;
  QD = Q.get_QD();
  clone(p, p_, l);
  clone(y, y_, l);
  clone(alpha, alpha_, l);
  this->Cp = Cp;
  this->Cn = Cn;
  this->eps = eps;
  unshrink = false;

  // initialize alpha_status
  {
    alpha_status = new char[l];
    for (int i = 0; i < l; i++) update_alpha_status(i);
  }

  // initialize active set (for shrinking)
  {
    active_set = new int[l];
    for (int i = 0; i < l; i++) active_set[i] = i;
    active_size = l;
  }

  // initialize gradient
  {
    G = new double[l];
    G_bar = new double[l];
    int i;
    for (i = 0; i < l; i++) {
      G[i] = p[i];
      G_bar[i] = 0;
    }
    for (i = 0; i < l; i++)
      if (!is_lower_bound(i)) {
        const Qfloat *Q_i = Q.get_Q(i, l);
        double alpha_i = alpha[i];
        int j;
        for (j = 0; j < l; j++) G[j] += alpha_i * Q_i[j];
        if (is_upper_bound(i))
          for (j = 0; j < l; j++) G_bar[j] += get_C(i) * Q_i[j];
      }
  }

  // optimization step

  int iter = 0;
  int counter = min(l, 1000) + 1;

  while (1) {
    // show progress and do shrinking

    if (--counter == 0) {
      counter = min(l, 1000);
      if (shrinking) do_shrinking();
    }

    int i, j;
    if (select_working_set(i, j) != 0 || iter > MAXSVMITERATION) {
      // reconstruct the whole gradient
      reconstruct_gradient();
      // reset active set size and check
      active_size = l;
      if (select_working_set(i, j) != 0 || iter > MAXSVMITERATION)
        break;
      else
        counter = 1;  // do shrinking next iteration
    }

    ++iter;

    // update alpha[i] and alpha[j], handle bounds carefully

    const Qfloat *Q_i = Q.get_Q(i, active_size);
    const Qfloat *Q_j = Q.get_Q(j, active_size);

    double C_i = get_C(i);
    double C_j = get_C(j);

    double old_alpha_i = alpha[i];
    double old_alpha_j = alpha[j];

    if (y[i] != y[j]) {
      double quad_coef = QD[i] + QD[j] + 2 * Q_i[j];
      if (quad_coef <= 0) quad_coef = TAU;
      double delta = (-G[i] - G[j]) / quad_coef;
      double diff = alpha[i] - alpha[j];
      alpha[i] += delta;
      alpha[j] += delta;

      if (diff > 0) {
        if (alpha[j] < 0) {
          alpha[j] = 0;
          alpha[i] = diff;
        }
      } else {
        if (alpha[i] < 0) {
          alpha[i] = 0;
          alpha[j] = -diff;
        }
      }
      if (diff > C_i - C_j) {
        if (alpha[i] > C_i) {
          alpha[i] = C_i;
          alpha[j] = C_i - diff;
        }
      } else {
        if (alpha[j] > C_j) {
          alpha[j] = C_j;
          alpha[i] = C_j + diff;
        }
      }
    } else {
      double quad_coef = QD[i] + QD[j] - 2 * Q_i[j];
      if (quad_coef <= 0) quad_coef = TAU;
      double delta = (G[i] - G[j]) / quad_coef;
      double sum = alpha[i] + alpha[j];
      alpha[i] -= delta;
      alpha[j] += delta;

      if (sum > C_i) {
        if (alpha[i] > C_i) {
          alpha[i] = C_i;
          alpha[j] = sum - C_i;
        }
      } else {
        if (alpha[j] < 0) {
          alpha[j] = 0;
          alpha[i] = sum;
        }
      }
      if (sum > C_j) {
        if (alpha[j] > C_j) {
          alpha[j] = C_j;
          alpha[i] = sum - C_j;
        }
      } else {
        if (alpha[i] < 0) {
          alpha[i] = 0;
          alpha[j] = sum;
        }
      }
    }

    // update G

    double delta_alpha_i = alpha[i] - old_alpha_i;
    double delta_alpha_j = alpha[j] - old_alpha_j;

    for (int k = 0; k < active_size; k++) {
      G[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j;
    }

    // update alpha_status and G_bar

    {
      bool ui = is_upper_bound(i);
      bool uj = is_upper_bound(j);
      update_alpha_status(i);
      update_alpha_status(j);
      int k;
      if (ui != is_upper_bound(i)) {
        Q_i = Q.get_Q(i, l);
        if (ui)
          for (k = 0; k < l; k++) G_bar[k] -= C_i * Q_i[k];
        else
          for (k = 0; k < l; k++) G_bar[k] += C_i * Q_i[k];
      }

      if (uj != is_upper_bound(j)) {
        Q_j = Q.get_Q(j, l);
        if (uj)
          for (k = 0; k < l; k++) G_bar[k] -= C_j * Q_j[k];
        else
          for (k = 0; k < l; k++) G_bar[k] += C_j * Q_j[k];
      }
    }
  }

  // calculate rho

  si->rho = calculate_rho();

  // calculate objective value
  {
    double v = 0;
    int i;
    for (i = 0; i < l; i++) v += alpha[i] * (G[i] + p[i]);

    si->obj = v / 2;
  }

  // put back the solution
  {
    for (int i = 0; i < l; i++) alpha_[active_set[i]] = alpha[i];
  }

  si->upper_bound_p = Cp;
  si->upper_bound_n = Cn;

  // printf("iteration: %d\n", iter);
  delete[] p;
  delete[] y;
  delete[] alpha;
  delete[] alpha_status;
  delete[] active_set;
  delete[] G;
  delete[] G_bar;
}

// return 1 if already optimal, return 0 otherwise
int Solver::select_working_set(int &out_i, int &out_j) {
  // return i,j such that
  // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
  // j: minimizes the decrease of obj value
  //    (if quadratic coefficeint <= 0, replace it with tau)
  //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

  ALIGNED(64) double Gmax = -INF;
  ALIGNED(64) double Gmax2 = -INF;
  int Gmax_idx = -1;
  int Gmin_idx = -1;
  ALIGNED(64) double obj_diff_min = INF;

  ALIGNED(64) double g1[active_size + 8];
  for (int t = 0; t < active_size; t++) {
    g1[t] = (y[t] == +1 && alpha_status[t] != UPPER_BOUND) ||
                    (y[t] == -1 && alpha_status[t] != LOWER_BOUND)
                ? -y[t] * G[t]
                : -INF;
  }
#ifdef __MIC__
  ALIGNED(64) double Gmax_arr[] = {Gmax, Gmax, Gmax, Gmax,
                                   Gmax, Gmax, Gmax, Gmax};
  __m512d cur_max = _mm512_load_pd((void const *)Gmax_arr);
  int t;
  for (t = active_size; t < active_size + 8; t++)  // padding
  {
    g1[t] = -INF;
  }
  for (t = 0; t < active_size; t += 8) {
    __m512d value = _mm512_load_pd((void const *)(g1 + t));
    cur_max = _mm512_gmax_pd(cur_max, value);
  }
  _mm512_store_pd((void *)Gmax_arr, cur_max);
#pragma loop count(8)
  for (t = 0; t < 8; t++) {
    if (Gmax < Gmax_arr[t]) Gmax = Gmax_arr[t];
  }
#pragma loop count(8)
  for (t = 0; t < 8; t++) {
    Gmax_arr[t] = Gmax;
  }
  cur_max = _mm512_load_pd((void const *)Gmax_arr);
  for (t = 0; t < active_size; t += 8) {
    __m512d value = _mm512_load_pd((void const *)(g1 + t));
    __mmask8 mask8 = _mm512_cmpeq_pd_mask(cur_max, value);
    int mask = _mm512_mask2int(mask8);
    if (mask) {
      int offset = 0;
      while ((mask & 0x1) == 0) {
        offset++;
        mask = mask >> 1;
      }
      Gmax_idx = t + offset;
      break;
    }
  }
#else
  for (int t = 0; t < active_size; t++) {
    if (g1[t] >= Gmax) {
      Gmax = g1[t];
      Gmax_idx = t;
    }
  }
#endif

  int i = Gmax_idx;
  const Qfloat *Q_i = NULL;
  // printf("%d ", active_size);
  if (i != -1)  // NULL Q_i not accessed: Gmax=-INF if i=-1
    Q_i = Q->get_Q(i, active_size);

  assert(Q_i);
  assert(active_size > 0);
  ALIGNED(64) double grad_diff[active_size];
  ALIGNED(64) double obj_diff[active_size + 8];
  ALIGNED(64) double g[active_size + 8];
#pragma simd
  for (int j = 0; j < active_size; j++) {
    grad_diff[j] = (y[j] == +1 && alpha_status[j] != LOWER_BOUND) ||
                           (y[j] == -1 && alpha_status[j] != UPPER_BOUND)
                       ? Gmax + G[j] * y[j]
                       : -1.0;
    g[j] = (y[j] == +1 && alpha_status[j] != LOWER_BOUND) ||
                   (y[j] == -1 && alpha_status[j] != UPPER_BOUND)
               ? G[j] * y[j]
               : -INF;
    ALIGNED(64) double quad_coef =
        QD[i] + QD[j] - 2.0 * y[i] * Q_i[j];  // y[i], but not y[j], is intended
    obj_diff[j] = quad_coef > 0 ? -(grad_diff[j] * grad_diff[j]) / quad_coef
                                : -(grad_diff[j] * grad_diff[j]) / TAU;
    obj_diff[j] = grad_diff[j] > 0 ? obj_diff[j] : INF;
  }
  for (int j = active_size; j < active_size + 8; j++)  // padding
  {
    g[j] = -INF;
    obj_diff[j] = INF;
  }
#ifdef __MIC__
  ALIGNED(64) double Gmax2_arr[] = {Gmax2, Gmax2, Gmax2, Gmax2,
                                    Gmax2, Gmax2, Gmax2, Gmax2};
  cur_max = _mm512_load_pd((void const *)Gmax2_arr);

  ALIGNED(64) double obj_diff_min_arr[] = {
      obj_diff_min, obj_diff_min, obj_diff_min, obj_diff_min,
      obj_diff_min, obj_diff_min, obj_diff_min, obj_diff_min};
  __m512d cur_min = _mm512_load_pd((void const *)obj_diff_min_arr);
  int j;
  for (j = 0; j < active_size; j += 8) {
    __m512d value = _mm512_load_pd((void const *)(g + j));
    cur_max = _mm512_gmax_pd(cur_max, value);

    value = _mm512_load_pd((void const *)(obj_diff + j));
    cur_min = _mm512_gmin_pd(cur_min, value);
  }
  _mm512_store_pd((void *)Gmax2_arr, cur_max);
  _mm512_store_pd((void *)obj_diff_min_arr, cur_min);
#pragma loop count(8)
  for (j = 0; j < 8; j++) {
    if (Gmax2 < Gmax2_arr[j]) Gmax2 = Gmax2_arr[j];
    if (obj_diff_min > obj_diff_min_arr[j]) obj_diff_min = obj_diff_min_arr[j];
  }
#pragma loop count(8)
  for (j = 0; j < 8; j++) {
    obj_diff_min_arr[j] = obj_diff_min;
  }
  cur_min = _mm512_load_pd((void const *)obj_diff_min_arr);
  for (j = 0; j < active_size; j += 8) {
    __m512d value = _mm512_load_pd((void const *)(obj_diff + j));
    __mmask8 mask8 = _mm512_cmpeq_pd_mask(cur_min, value);
    int mask = _mm512_mask2int(mask8);
    if (mask) {
      int offset = 0;
      while ((mask & 0x1) == 0) {
        offset++;
        mask = mask >> 1;
      }
      Gmin_idx = j + offset;
      break;
    }
  }
#else
  for (int j = 0; j < active_size; j++) {
    if (g[j] >= Gmax2) Gmax2 = g[j];
    if (obj_diff[j] <= obj_diff_min) {
      Gmin_idx = j;
      obj_diff_min = obj_diff[j];
    }
  }
#endif
  if (Gmax + Gmax2 < eps) return 1;

  out_i = Gmax_idx;
  out_j = Gmin_idx;
  return 0;
}

bool Solver::be_shrunk(int i, double Gmax1, double Gmax2) {
  if (is_upper_bound(i)) {
    if (y[i] == +1)
      return (-G[i] > Gmax1);
    else
      return (-G[i] > Gmax2);
  } else if (is_lower_bound(i)) {
    if (y[i] == +1)
      return (G[i] > Gmax2);
    else
      return (G[i] > Gmax1);
  } else
    return (false);
}

void Solver::do_shrinking() {
  int i;
  double Gmax1 = -INF;  // max { -y_i * grad(f)_i | i in I_up(\alpha) }
  double Gmax2 = -INF;  // max { y_i * grad(f)_i | i in I_low(\alpha) }

  // find maximal violating pair first
  for (i = 0; i < active_size; i++) {
    if (y[i] == +1) {
      if (!is_upper_bound(i)) {
        if (-G[i] >= Gmax1) Gmax1 = -G[i];
      }
      if (!is_lower_bound(i)) {
        if (G[i] >= Gmax2) Gmax2 = G[i];
      }
    } else {
      if (!is_upper_bound(i)) {
        if (-G[i] >= Gmax2) Gmax2 = -G[i];
      }
      if (!is_lower_bound(i)) {
        if (G[i] >= Gmax1) Gmax1 = G[i];
      }
    }
  }

  if (unshrink == false && Gmax1 + Gmax2 <= eps * 10) {
    unshrink = true;
    reconstruct_gradient();
    active_size = l;
  }

  for (i = 0; i < active_size; i++)
    if (be_shrunk(i, Gmax1, Gmax2)) {
      active_size--;
      while (active_size > i) {
        if (!be_shrunk(active_size, Gmax1, Gmax2)) {
          swap_index(i, active_size);
          break;
        }
        active_size--;
      }
    }
}
double Solver::calculate_rho() {
  double r;
  int nr_free = 0;
  double ub = INF, lb = -INF, sum_free = 0;
  for (int i = 0; i < active_size; i++) {
    double yG = y[i] * G[i];

    if (is_upper_bound(i)) {
      if (y[i] == -1)
        ub = min(ub, yG);
      else
        lb = max(lb, yG);
    } else if (is_lower_bound(i)) {
      if (y[i] == +1)
        ub = min(ub, yG);
      else
        lb = max(lb, yG);
    } else {
      ++nr_free;
      sum_free += yG;
    }
  }

  if (nr_free > 0)
    r = sum_free / nr_free;
  else
    r = (ub + lb) / 2;

  return r;
}

//
// Q matrices for various formulations
//
class SVC_Q : public Kernel {
 public:
  SVC_Q(const svm_problem &prob, const svm_parameter &param, const schar *y_)
      : Kernel(prob.l, prob.x, param) {
    clone(y, y_, prob.l);
    cache = new Cache(prob.l,
                      (long int)(param.cache_size * (1 << 20)));  // 1<<20->1MB
    QD = new double[prob.l];
    for (int i = 0; i < prob.l; i++) QD[i] = (this->*kernel_function)(i, i);
  }

  Qfloat *get_Q(int i, int len) const {
    Qfloat *data;
    int start, j;
    if ((start = cache->get_data(i, &data, len)) < len) {
      for (j = start; j < len; j++)
        data[j] = (Qfloat)(y[i] * y[j] * (this->*kernel_function)(i, j));
    }
    return data;
  }

  double *get_QD() const { return QD; }

  void swap_index(int i, int j) const {
    cache->swap_index(i, j);
    Kernel::swap_index(i, j);
    swap(y[i], y[j]);
    swap(QD[i], QD[j]);
  }

  ~SVC_Q() {
    delete[] y;
    delete cache;
    delete[] QD;
  }

 private:
  schar *y;
  Cache *cache;
  double *QD;
};

//
// construct and solve various formulations
//
static void solve_c_svc(const svm_problem *prob, const svm_parameter *param,
                        double *alpha, Solver::SolutionInfo *si, double Cp,
                        double Cn) {
  int l = prob->l;
  double *minus_ones = new double[l];
  schar *y = new schar[l];

  int i;

  for (i = 0; i < l; i++) {
    alpha[i] = 0;
    minus_ones[i] = -1;
    if (prob->y[i] > 0)
      y[i] = +1;
    else
      y[i] = -1;
  }

  Solver s;
  s.Solve(l, SVC_Q(*prob, *param, y), minus_ones, y, alpha, Cp, Cn, param->eps,
          si, param->shrinking);

  double sum_alpha = 0;
  for (i = 0; i < l; i++) sum_alpha += alpha[i];

  for (i = 0; i < l; i++) alpha[i] *= y[i];

  delete[] minus_ones;
  delete[] y;
}

//
// decision_function
//
struct decision_function {
  double *alpha;
  double rho;
};

static decision_function svm_train_one(const svm_problem *prob,
                                       const svm_parameter *param, double Cp,
                                       double Cn) {
  double *alpha = Calloc(double, prob->l);
  Solver::SolutionInfo si;
  memset(&si, 0, sizeof(Solver::SolutionInfo));
  switch (param->svm_type) {
    case C_SVC:
      solve_c_svc(prob, param, alpha, &si, Cp, Cn);
      break;
  }

  // output SVs

  int nSV = 0;
  int nBSV = 0;
  for (int i = 0; i < prob->l; i++) {
    if (fabs(alpha[i]) > 0) {
      ++nSV;
      if (prob->y[i] > 0) {
        if (fabs(alpha[i]) >= si.upper_bound_p) ++nBSV;
      } else {
        if (fabs(alpha[i]) >= si.upper_bound_n) ++nBSV;
      }
    }
  }

  decision_function f;
  f.alpha = alpha;
  f.rho = si.rho;
  return f;
}

// label: label name, start: begin of each class, count: #data of classes, perm:
// indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void svm_group_classes(const svm_problem *prob, int *nr_class_ret,
                              int **label_ret, int **start_ret, int **count_ret,
                              int *perm) {
  int l = prob->l;
  int max_nr_class = 16;
  int nr_class = 0;
  int *label = Calloc(int, max_nr_class);
  int *count = Calloc(int, max_nr_class);
  int *data_label = Calloc(int, l);
  int i;

  for (i = 0; i < l; i++) {
    int this_label = (int)prob->y[i];
    int j;
    for (j = 0; j < nr_class; j++) {
      if (this_label == label[j]) {
        ++count[j];
        break;
      }
    }
    data_label[i] = j;
    if (j == nr_class) {
      if (nr_class == max_nr_class) {
        max_nr_class *= 2;
        label = (int *)realloc(label, max_nr_class * sizeof(int));
        count = (int *)realloc(count, max_nr_class * sizeof(int));
      }
      label[nr_class] = this_label;
      count[nr_class] = 1;
      ++nr_class;
    }
  }
  assert(nr_class > 0);
  int *start = Calloc(int, nr_class);
  start[0] = 0;
  for (i = 1; i < nr_class; i++) start[i] = start[i - 1] + count[i - 1];
  for (i = 0; i < l; i++) {
    perm[start[data_label[i]]] = i;
    ++start[data_label[i]];
  }
  start[0] = 0;
  for (i = 1; i < nr_class; i++) start[i] = start[i - 1] + count[i - 1];

  *nr_class_ret = nr_class;
  *label_ret = label;
  *start_ret = start;
  *count_ret = count;
  free(data_label);
}
//
// Interface functions
//
svm_model *svm_train(const svm_problem *prob, const svm_parameter *param) {
  svm_model *model = Calloc(svm_model, 1);
  model->param = *param;
  model->free_sv = 0;  // XXX

  // classification
  int l = prob->l;
  int nr_class;
  int *label = NULL;
  int *start = NULL;
  int *count = NULL;
  int *perm = Calloc(int, l);

  // group training data of the same class
  svm_group_classes(prob, &nr_class, &label, &start, &count, perm);
  svm_node **x = Calloc(svm_node *, l);
  int i;
  for (i = 0; i < l; i++) x[i] = prob->x[perm[i]];

  // calculate weighted C

  double *weighted_C = Calloc(double, nr_class);
  for (i = 0; i < nr_class; i++) weighted_C[i] = param->C;
  for (i = 0; i < param->nr_weight; i++) {
    int j;
    for (j = 0; j < nr_class; j++)
      if (param->weight_label[i] == label[j]) break;
    if (j == nr_class)
      fprintf(stderr,
              "warning: class label %d specified in weight is not found\n",
              param->weight_label[i]);
    else
      weighted_C[j] *= param->weight[i];
  }

  // train k*(k-1)/2 models

  bool *nonzero = Calloc(bool, l);
  for (i = 0; i < l; i++) nonzero[i] = false;
  assert(nr_class > 1);
  decision_function *f =
      Calloc(decision_function, nr_class * (nr_class - 1) / 2);

  double *probA = NULL, *probB = NULL;

  int p = 0;
  for (i = 0; i < nr_class; i++)
    for (int j = i + 1; j < nr_class; j++) {
      svm_problem sub_prob;
      int si = start[i], sj = start[j];
      int ci = count[i], cj = count[j];
      sub_prob.l = ci + cj;
      sub_prob.x = Calloc(svm_node *, sub_prob.l);
      sub_prob.y = Calloc(schar, sub_prob.l);

      int k = 0;
      for (k = 0; k < ci; k++) {
        sub_prob.x[k] = x[si + k];
        sub_prob.y[k] = +1;
      }
      for (k = 0; k < cj; k++) {
        sub_prob.x[ci + k] = x[sj + k];
        sub_prob.y[ci + k] = -1;
      }

      f[p] = svm_train_one(&sub_prob, param, weighted_C[i], weighted_C[j]);
      for (k = 0; k < ci; k++)
        if (!nonzero[si + k] && fabs(f[p].alpha[k]) > 0) nonzero[si + k] = true;
      for (k = 0; k < cj; k++)
        if (!nonzero[sj + k] && fabs(f[p].alpha[ci + k]) > 0)
          nonzero[sj + k] = true;
      free(sub_prob.x);
      free(sub_prob.y);
      ++p;
    }

  // build output

  model->nr_class = nr_class;

  model->label = Calloc(int, nr_class);
  for (i = 0; i < nr_class; i++) model->label[i] = label[i];

  model->rho = Calloc(double, nr_class * (nr_class - 1) / 2);
  for (i = 0; i < nr_class * (nr_class - 1) / 2; i++) model->rho[i] = f[i].rho;

  int total_sv = 0;
  int *nz_count = Calloc(int, nr_class);
  model->nSV = Calloc(int, nr_class);
  for (i = 0; i < nr_class; i++) {
    int nSV = 0;
    for (int j = 0; j < count[i]; j++)
      if (nonzero[start[i] + j]) {
        ++nSV;
        ++total_sv;
      }
    model->nSV[i] = nSV;
    nz_count[i] = nSV;
  }

  assert(total_sv > 0);
  model->l = total_sv;
  model->SV = Calloc(svm_node *, total_sv);
  p = 0;
  for (i = 0; i < l; i++)
    if (nonzero[i]) model->SV[p++] = x[i];

  int *nz_start = Calloc(int, nr_class);
  nz_start[0] = 0;
  for (i = 1; i < nr_class; i++)
    nz_start[i] = nz_start[i - 1] + nz_count[i - 1];

  model->sv_coef = Calloc(double *, nr_class - 1);
  for (i = 0; i < nr_class - 1; i++)
    model->sv_coef[i] = Calloc(double, total_sv);

  p = 0;
  for (i = 0; i < nr_class; i++)
    for (int j = i + 1; j < nr_class; j++) {
      // classifier (i,j): coefficients with
      // i are in sv_coef[j-1][nz_start[i]...],
      // j are in sv_coef[i][nz_start[j]...]

      int si = start[i];
      int sj = start[j];
      int ci = count[i];
      int cj = count[j];

      int q = nz_start[i];
      int k;
      for (k = 0; k < ci; k++)
        if (nonzero[si + k]) model->sv_coef[j - 1][q++] = f[p].alpha[k];
      q = nz_start[j];
      for (k = 0; k < cj; k++)
        if (nonzero[sj + k]) model->sv_coef[i][q++] = f[p].alpha[ci + k];
      ++p;
    }

  free(label);
  free(probA);
  free(probB);
  free(count);
  free(perm);
  free(start);
  free(x);
  free(weighted_C);
  free(nonzero);
  for (i = 0; i < nr_class * (nr_class - 1) / 2; i++) free(f[i].alpha);
  free(f);
  free(nz_count);
  free(nz_start);
  return model;
}

// stratified cross validation no random shuffle -- Yida
void svm_cross_validation_no_shuffle(const svm_problem *prob,
                                     const svm_parameter *param, int nr_fold,
                                     double *target) {
  int i;
  int *fold_start = Calloc(int, nr_fold + 1);
  int l = prob->l;
  int *perm = Calloc(int, l);
  int nr_class;
  // stratified cv may not give leave-one-out rate
  // Each class to l folds -> some folds may have zero elements
  if (param->svm_type == C_SVC && nr_fold < l) {
    int *start = NULL;
    int *label = NULL;
    int *count = NULL;
    // put the same class to be adjacent to each other
    svm_group_classes(prob, &nr_class, &label, &start, &count, perm);

    // data grouped by fold using the array perm
    int *fold_count = Calloc(int, nr_fold);
    int c;
    int *index = Calloc(int, l);
    for (i = 0; i < l; i++) index[i] = perm[i];
    // according to the number of folds, assign each fold with same number of
    // different classes
    for (i = 0; i < nr_fold; i++) {
      fold_count[i] = 0;
      for (c = 0; c < nr_class; c++)
        fold_count[i] += (i + 1) * count[c] / nr_fold - i * count[c] / nr_fold;
    }
    fold_start[0] = 0;
    for (i = 1; i <= nr_fold; i++)
      fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
    for (c = 0; c < nr_class; c++)
      for (i = 0; i < nr_fold; i++) {
        int begin = start[c] + i * count[c] / nr_fold;
        int end = start[c] + (i + 1) * count[c] / nr_fold;
        for (int j = begin; j < end; j++) {
          perm[fold_start[i]] = index[j];
          fold_start[i]++;
        }
      }
    fold_start[0] = 0;
    for (i = 1; i <= nr_fold; i++)
      fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
    free(start);
    free(label);
    free(count);
    free(index);
    free(fold_count);
  }
  for (i = 0; i < nr_fold; i++) {
    int begin = fold_start[i];
    int end = fold_start[i + 1];
    int j, k;
    struct svm_problem subprob;

    subprob.l = l - (end - begin);
    subprob.x = Calloc(struct svm_node *, subprob.l);
    subprob.y = Calloc(schar, subprob.l);

    k = 0;
    for (j = 0; j < begin; j++) {
      subprob.x[k] = prob->x[perm[j]];
      subprob.y[k] = prob->y[perm[j]];
      ++k;
    }
    for (j = end; j < l; j++) {
      subprob.x[k] = prob->x[perm[j]];
      subprob.y[k] = prob->y[perm[j]];
      ++k;
    }
    struct svm_model *submodel = svm_train(&subprob, param);
    for (j = begin; j < end; j++)
      target[perm[j]] = svm_predict(submodel, prob->x[perm[j]]);
    svm_free_and_destroy_model(&submodel);
    free(subprob.x);
    free(subprob.y);
  }
  free(fold_start);
  free(perm);
}

int svm_get_svm_type(const svm_model *model) { return model->param.svm_type; }

int svm_get_nr_class(const svm_model *model) { return model->nr_class; }

void svm_get_labels(const svm_model *model, int *label) {
  if (model->label != NULL)
    for (int i = 0; i < model->nr_class; i++) label[i] = model->label[i];
}

double svm_predict_values(const svm_model *model, const svm_node *x,
                          double *dec_values) {
  int length = 0;
  const svm_node *temp_x = x;
  while (temp_x->index != -1) {
    length++;
    temp_x++;
  }
  int i;
  int nr_class = model->nr_class;
  int l = model->l;

  double *kvalue = Calloc(double, l);
#pragma omp parallel for private(i)
  for (i = 0; i < l; i++)
    kvalue[i] = Kernel::k_function(x, model->SV[i], model->param, length);

  int *start = Calloc(int, nr_class);
  start[0] = 0;
  for (i = 1; i < nr_class; i++) start[i] = start[i - 1] + model->nSV[i - 1];

  int *vote = Calloc(int, nr_class);
  for (i = 0; i < nr_class; i++) vote[i] = 0;

  int p = 0;
  for (i = 0; i < nr_class; i++)
    for (int j = i + 1; j < nr_class; j++) {
      double sum = 0;
      int si = start[i];
      int sj = start[j];
      int ci = model->nSV[i];
      int cj = model->nSV[j];

      int k;
      double *coef1 = model->sv_coef[j - 1];
      double *coef2 = model->sv_coef[i];
      for (k = 0; k < ci; k++) sum += coef1[si + k] * kvalue[si + k];
      for (k = 0; k < cj; k++) sum += coef2[sj + k] * kvalue[sj + k];
      sum -= model->rho[p];
      dec_values[p] = sum;

      if (dec_values[p] > 0)
        ++vote[i];
      else
        ++vote[j];
      p++;
    }

  int vote_max_idx = 0;
  for (i = 1; i < nr_class; i++)
    if (vote[i] > vote[vote_max_idx]) vote_max_idx = i;

  free(kvalue);
  free(start);
  free(vote);
  return model->label[vote_max_idx];
}

double svm_predict(const svm_model *model, const svm_node *x) {
  int nr_class = model->nr_class;
  double *dec_values;
  dec_values = Calloc(double, nr_class * (nr_class - 1) / 2);
  double pred_result = svm_predict_values(model, x, dec_values);
  free(dec_values);
  return pred_result;
}

// output the distance to the hyperplane, now only deal with 2-class
// classification
double svm_predict_distance(const struct svm_model *model,
                            const struct svm_node *x) {
  int nr_class = model->nr_class;
  double *dec_values;
  dec_values = Calloc(double, nr_class * (nr_class - 1) / 2);
  svm_predict_values(model, x, dec_values);
  double pred_distance = dec_values[0];
  free(dec_values);
  return pred_distance;
}

void svm_free_model_content(svm_model *model_ptr) {
  if (model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL)
    free((void *)(model_ptr->SV[0]));
  if (model_ptr->sv_coef) {
    for (int i = 0; i < model_ptr->nr_class - 1; i++)
      free(model_ptr->sv_coef[i]);
  }

  free(model_ptr->SV);
  model_ptr->SV = NULL;

  free(model_ptr->sv_coef);
  model_ptr->sv_coef = NULL;

  free(model_ptr->rho);
  model_ptr->rho = NULL;

  free(model_ptr->label);
  model_ptr->label = NULL;

  free(model_ptr->nSV);
  model_ptr->nSV = NULL;
}

void svm_free_and_destroy_model(svm_model **model_ptr_ptr) {
  if (model_ptr_ptr != NULL && *model_ptr_ptr != NULL) {
    svm_free_model_content(*model_ptr_ptr);
    free(*model_ptr_ptr);
    *model_ptr_ptr = NULL;
  }
}

void svm_destroy_param(svm_parameter *param) {
  free(param->weight_label);
  free(param->weight);
}

const char *svm_check_parameter(const svm_problem *prob,
                                const svm_parameter *param) {
  // svm_type

  int svm_type = param->svm_type;
  if (svm_type != C_SVC) return "unknown svm type";

  // kernel_type, degree

  int kernel_type = param->kernel_type;
  if (kernel_type != LINEAR && kernel_type != PRECOMPUTED)
    return "unknown kernel type";

  if (param->gamma < 0) return "gamma < 0";

  if (param->degree < 0) return "degree of polynomial kernel < 0";

  // cache_size,eps,C,nu,p,shrinking

  if (param->cache_size <= 0) return "cache_size <= 0";

  if (param->eps <= 0) return "eps <= 0";

  if (svm_type == C_SVC)
    if (param->C <= 0) return "C <= 0";

  if (param->shrinking != 0 && param->shrinking != 1)
    return "shrinking != 0 and shrinking != 1";

  return NULL;
}

static void print_string_stdout(const char *s) {
  fputs(s, stdout);
  fflush(stdout);
}
static void (*svm_print_string)(const char *) = &print_string_stdout;

void svm_set_print_string_function(void (*print_func)(const char *)) {
  svm_print_string = print_func;
}
