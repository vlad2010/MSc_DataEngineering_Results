#include <arm_neon.h>
#include <omp.h>

double dot_product(float* a, float* b, int size) {
    float result_array[4];
    double result = 0.0;

    #pragma omp parallel
    {
        float32x4_t a_vec, b_vec, result_vec = vdupq_n_f32(0.0f);

        #pragma omp for
        for(int i = 0; i < size; i += 4) {
            a_vec = vld1q_f32(a + i);
            b_vec = vld1q_f32(b + i);
            result_vec = vmlaq_f32(result_vec, a_vec, b_vec);
        }

        vst1q_f32(result_array, result_vec);

        #pragma omp atomic
        for(int i = 0; i < 4; i++) {
            result += result_array[i];
        }
    }

    return result;
}
