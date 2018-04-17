#ifndef LIB_HPP
#define LIB_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#define len(a, b) (int (a)=0; (a) < static_cast<int>((b).size()); ++(a))
#define range(a, b) (int (a)=0; (a) < (b); ++(a))

template <typename T>
using Mat1D = std::vector<T>;

template <typename T>
using Mat2D = std::vector< std::vector<T> >;

template <typename T>
using Mat3D = std::vector< std::vector< std::vector<T> > >;

template <typename T>
using Mat4D = std::vector< std::vector< std::vector< std::vector<T> > > >;

template <typename T>
Mat1D<T> zeros(int size1)
{
  Mat1D<T> target;

  target.resize(size1);

  return target;
}

template <typename T>
Mat2D<T> zeros(int size1, int size2)
{
  Mat2D<T> target;

  target.resize(size1);
  for range(i, size1)
    target[i].resize(size2);

  return target;
}

template <typename T>
Mat3D<T> zeros(int size1, int size2, int size3)
{
  Mat3D<T> target;

  target.resize(size1);
  for range(i, size1) {
    target[i].resize(size2);
    for range(j, size2)
      target[i][j].resize(size3);
  }

  return target;
}

template <typename T>
Mat4D<T> zeros(int size1, int size2, int size3, int size4)
{
  Mat4D<T> target;

  target.resize(size1);
  for range(i, size1) {
    target[i].resize(size2);
    for range(j, size2) {
      target[i][j].resize(size3);
      for range(k, size3)
        target[i][j][k].resize(size4);
    }
  }

  return target;
}

template <typename T>
void load(Mat1D<T> &x, std::string path)
{
  std::ifstream ifs(path);

  for len(i, x) {
    ifs >> x[i];
  }
}

template <typename T>
void load(Mat2D<T> &x, std::string path)
{
  std::ifstream ifs(path);

  for len(i, x)
    for len(j, x[0]) {
      ifs >> x[i][j];
  }
}

template <typename T>
void load(Mat3D<T> &x, std::string path)
{
  std::ifstream ifs(path);

  for len(i, x)
    for len(j, x[0])
      for len(k, x[0][0]) {
        ifs >> x[i][j][k];
  }
}

template <typename T>
void load(Mat4D<T> &x, std::string path)
{
  std::ifstream ifs(path);

  for len(i, x)
    for len(j, x[0])
      for len(k, x[0][0])
        for len(l, x[0][0][0]) {
          ifs >> x[i][j][k][l];
  }
}

template <typename T>
void save(Mat1D<T> &y, std::string path)
{
  std::ofstream ofs(path);

  for len(i, y)
    ofs << y[i] << std::endl;
}

template <typename T>
void save(Mat2D<T> &y, std::string path)
{
  std::ofstream ofs(path);

  for len(i, y)
    for len(j, y[0])
      ofs << y[i][j] << std::endl;
}

template <typename T>
void save(Mat3D<T> &y, std::string path)
{
  std::ofstream ofs(path);

  for len(i, y)
    for len(j, y[0])
      for len(k, y[0][0])
        ofs << y[i][j][k] << std::endl;
}

template <typename T>
void save(Mat4D<T> &y, std::string path)
{
  std::ofstream ofs(path);

  for len(i, y)
    for len(j, y[0])
      for len(k, y[0][0])
        for len(l, y[0][0][0])
          ofs << y[i][j][k][l] << std::endl;
}

#endif
