#include <cmath>
#include <algorithm>

using namespace std;

void get_trend(const double *f,
               const bool *ol,
               const int arr_size,
               const int kernel_size,
               double *f_trend,
               double *f_error)
{
  /*
  Return the moving window mean and standard error.
  Arguments:
  -- array (ordered)
  -- boolean mask, true where element should be omitted from mean/stderr
  -- array length
  -- kernel size (number of points each side array elements to define window)
  Returns:
  -- mean in window around array element
  -- standard error on mean in window around array element
  */

  // make the measurement
  int jstart, jend, count = 0;
  float sum = 0.0, ssqd = 0.0;
  for (int i = 0; i < arr_size; i++) // for each point
  {
    jstart = max(0, i-kernel_size);
    jend = min(arr_size, i+kernel_size);
    for (int j = jstart; j < jend; j++) // for each point in window
    {
      if (!ol[j]) {
        sum += f[j];
        count += 1;
      }
    }
    f_trend[i] = sum/count;

    for (int j = jstart; j < jend; j++) // for each point in window
    {
      if (!ol[j]) {
        ssqd += pow(f[j]-f_trend[i], 2);
      }
    }
    f_error[i] = sqrt(ssqd / count) / sqrt(count);

    // reset counters
    sum = 0.0;
    ssqd = 0.0;
    count = 0;
  }

}

extern "C" {
  void c_get_trend(const double *f,
                   const bool *ol,
                   const int arr_size,
                   const int kernel_size,
                   double *f_trend,
                   double *f_error)
  {
      get_trend(f, ol, arr_size, kernel_size, f_trend, f_error);
  }
}
