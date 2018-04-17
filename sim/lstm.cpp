#include <iostream>
#include <cmath>
#include <cstdint>
#include <cassert>

#include "lib.hpp"

const int ttime = 5000;
const int n_in  = 1;
const int h_in  = 5;
const int h_out = 3*h_in;
const int n_out = 1;

template <typename T1, typename T2>
inline T1 decr(T2 x)
{
  if (x >= 0)
    return x / static_cast<T1>(pow(2, 8));
  else
    return x / static_cast<T1>(pow(2, 8)) - 1;
}

template <typename T1, typename T2>
inline T1 incr(T2 x)
{
  return x * static_cast<T1>(pow(2, 8));
}

inline float mul(float x, float y)
{
  return x * y;
}

inline int mul(int x, int y)
{
  return decr<int, int>(x * y);
}

void tanh(Mat1D<float> &y, Mat1D<float> &x)
{
  assert(y.size() == x.size());

  for len(i, y)
    y[i] = tanh(x[i]);
}

void tanh(Mat1D<int> &y, Mat1D<int> &x)
{
  assert(y.size() == x.size());

  for len(i, y)
    y[i] = incr<int, float>(tanh(decr<float, int>(x[i])));
}

void sigmoid(Mat1D<float> &y, Mat1D<float> &x)
{
  assert(y.size() == x.size());

  float half = 0.5;

  for len(i, y)
    y[i] = tanh(x[i] * half) * half + half;
}

void sigmoid(Mat1D<int> &y, Mat1D<int> &x)
{
  assert(y.size() == x.size());

  int half = 128;

  for len(i, y)
    y[i] = mul(incr<int, float>(tanh(decr<float, int>(mul(x[i], half)))), half) + half;
}

template <typename T>
void full(Mat1D<T> &y, Mat1D<T> &x, Mat2D<T> &W)
{
  assert(y.size() == W.size());
  assert(x.size() == W[0].size());

  for len(n, y) {
    y[n] = 0;
    for len(m, x) {
      y[n] += mul(W[n][m], x[m]);
    }
  }
}

template <typename T>
void bias(Mat1D<T> &y, Mat1D<T> &x, Mat1D<T> b)
{
  assert(y.size() == x.size());
  assert(x.size() == b.size());

  for len(n, y)
    y[n] = x[n] + b[n];
}

template <typename T>
void extract(Mat1D<T> &a, Mat1D<T> &i, Mat1D<T> &f, Mat1D<T> &o, Mat1D<T> &x)
{
  assert(a.size() == i.size());
  assert(f.size() == o.size());
  assert(i.size() == o.size());
  assert(x.size() == 4*a.size());

  for len(idx, f) {
    a[idx] = x[4*idx+0];
    i[idx] = x[4*idx+1];
    f[idx] = x[4*idx+2];
    o[idx] = x[4*idx+3];
  }
}

template <typename T>
void lstm(Mat1D<T> &y, Mat1D<T> &cn, Mat1D<T> &x, Mat1D<T> &cp, Mat1D<T> &h,
          Mat2D<T> &Wx, Mat1D<T> &bx, Mat2D<T> &Wh)
{
  assert(x.size()  == Wx[0].size());
  assert(y.size()  == cn.size());
  assert(cn.size() == cp.size());
  assert(Wx.size() == 4*cp.size());
  assert(Wx.size() == bx.size());
  assert(Wh.size() == 4*Wh[0].size());
  assert(Wx.size() == Wh.size());

  const int n_in  = x.size();
  const int n_hid = cn.size();

  auto xx = zeros<T>(4*n_hid);
  auto xh = zeros<T>(4*n_hid);
  auto hh = zeros<T>(4*n_hid);
  auto yh = zeros<T>(4*n_hid);

  full(xx, x, Wx);
  bias(xh, xx, bx);
  full(hh, h, Wh);

  for len(idx, xh)
    yh[idx] = xh[idx] + hh[idx];

  auto a = zeros<T>(n_hid);
  auto i = zeros<T>(n_hid);
  auto f = zeros<T>(n_hid);
  auto o = zeros<T>(n_hid);

  extract(a, i, f, o, yh);

  auto aa = zeros<T>(n_hid);
  auto ai = zeros<T>(n_hid);
  auto af = zeros<T>(n_hid);
  auto ao = zeros<T>(n_hid);

  tanh(aa, a);
  sigmoid(ai, i);
  sigmoid(af, f);
  sigmoid(ao, o);

  for len(idx, cn)
    cn[idx] = mul(aa[idx], ai[idx]) + mul(af[idx], cp[idx]);

  auto acn = zeros<T>(n_hid);

  tanh(acn, cn);

  for len(idx, y)
    y[idx] = mul(ao[idx], acn[idx]);
}

int main(void)
{
  using T = float;

  auto input    = zeros<T>(n_in);
  auto x_full0  = zeros<T>(h_in);
  auto y_full0  = zeros<T>(h_in);
  auto y_lstm1  = zeros<T>(h_out);
  auto h_lstm1  = zeros<T>(h_out);
  auto cp_lstm1 = zeros<T>(h_out);
  auto cn_lstm1 = zeros<T>(h_out);
  auto x_full2  = zeros<T>(n_out);
  auto y_full2  = zeros<T>(n_out);
  auto output   = zeros<T>(ttime, n_out);

  auto W_full0  = zeros<T>(h_in, n_in);
  auto b_full0  = zeros<T>(h_in);

  auto Wx_lstm1 = zeros<T>(4*h_out, h_in);
  auto bx_lstm1 = zeros<T>(4*h_out);
  auto Wh_lstm1 = zeros<T>(4*h_out, h_out);

  auto W_full2  = zeros<T>(n_out, h_out);
  auto b_full2  = zeros<T>(n_out);

  load(W_full0,  "rnn/full0/W.dat");
  load(b_full0,  "rnn/full0/b.dat");
  load(Wx_lstm1, "rnn/lstm1/upward/W.dat");
  load(bx_lstm1, "rnn/lstm1/upward/b.dat");
  load(Wh_lstm1, "rnn/lstm1/lateral/W.dat");
  load(W_full2,  "rnn/full2/W.dat");
  load(b_full2,  "rnn/full2/b.dat");

  // load(W_full0,  "data/W_full0.dat");
  // load(b_full0,  "data/b_full0.dat");
  // load(Wx_lstm1, "data/Wx_lstm1.dat");
  // load(bx_lstm1, "data/bx_lstm1.dat");
  // load(Wh_lstm1, "data/Wh_lstm1.dat");
  // load(W_full2,  "data/W_full2.dat");
  // load(b_full2,  "data/b_full2.dat");

  for range(t, ttime) {
    full(x_full0, input, W_full0);
    bias(y_full0, x_full0, b_full0);

    lstm(y_lstm1, cn_lstm1,
         y_full0, cp_lstm1, h_lstm1,
         Wx_lstm1, bx_lstm1, Wh_lstm1);
    cp_lstm1 = cn_lstm1;
    h_lstm1 = y_lstm1;

    full(x_full2, y_lstm1, W_full2);
    bias(y_full2, x_full2, b_full2);

    input = y_full2;
    output[t] = y_full2;
  }

  save(output, "output.dat");

  return 0;
}
