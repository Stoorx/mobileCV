#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <arm_neon.h>
#include <chrono>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

//void rgb_to_gray(const uint8_t* rgb, uint8_t* gray, int num_pixels)
//{
//    cout << "inside function rgb_to_gray" << endl;
//    auto t1 = chrono::high_resolution_clock::now();
//    for(int i=0; i<num_pixels; ++i, rgb+=3) {
//        int v = (77*rgb[0] + 150*rgb[1] + 29*rgb[2]);
//        gray[i] = v>>8;
//    }
//    auto t2 = chrono::high_resolution_clock::now();
//    auto duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();
//    cout << duration << " us" << endl;
//}
//
//void rgb_to_gray_neon(const uint8_t* rgb, uint8_t* gray, int num_pixels) {
//    // We'll use 64-bit NEON registers to process 8 pixels in parallel.
//    num_pixels /= 8;
//    // Duplicate the weight 8 times.
//    uint8x8_t w_r = vdup_n_u8(77);
//    uint8x8_t w_g = vdup_n_u8(150);
//    uint8x8_t w_b = vdup_n_u8(29);
//    // For intermediate results. 16-bit/pixel to avoid overflow.
//    uint16x8_t temp;
//    // For the converted grayscale values.
//    uint8x8_t result;
//    auto t1_neon = chrono::high_resolution_clock::now();
//    for(int i=0; i<num_pixels; ++i, rgb+=8*3, gray+=8) {
//        // Load 8 pixels into 3 64-bit registers, split by channel.
//        uint8x8x3_t src = vld3_u8(rgb);
//        // Multiply all eight red pixels by the corresponding weights.
//        temp = vmull_u8(src.val[0], w_r);
//        // Combined multiply and addition.
//        temp = vmlal_u8(temp, src.val[1], w_g);
//        temp = vmlal_u8(temp, src.val[2], w_b);
//        // Shift right by 8, "narrow" to 8-bits (recall temp is 16-bit).
//        result = vshrn_n_u16(temp, 8);
//        // Store converted pixels in the output grayscale image.
//        vst1_u8(gray, result);
//    }
//
//    auto t2_neon = chrono::high_resolution_clock::now();
//    auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
//    cout << "inside function rgb_to_gray_neon" << endl;
//    cout << duration_neon << " us" << endl;
//}

void rgb_to_hsv(const uint8_t* rgb, uint8_t* hsv, int num_pixels)
{
    num_pixels /= 8;
    uint8x8x3_t result;
    auto t1_neon = chrono::high_resolution_clock::now();
    auto zero = vdup_n_u8(0);
    for(int i=0; i<num_pixels; ++i, rgb+=8*3, hsv+=8*3) {
        // Load 8 pixels into 3 64-bit registers, split by channel.
        uint8x8x3_t src = vld3_u8(rgb);
        uint8x8_t max = vmax_u8(src.val[0], vmax_u8(src.val[1], src.val[2]));
        uint8x8_t min = vmin_u8(src.val[0], vmin_u8(src.val[1], src.val[2]));
        uint8x8_t delta = vsub_u8(max, min);
        uint8x8_t delta_positive_mask = vcgt_u8(delta, zero);
        for (size_t j = 0; j < 3; ++j) {
            auto color_mask = vceq_u8(max, src.val[j]);
            
            size_t i1 = (j + 1) % 3, i2 = (j + 2) % 3;

            uint8_t mult = 2 * j;
            uint16x8_t intermediate_result;
            uint8x8_t sub = vsub_u8(src.val[i1], src.val[i2]);
            if (j == 0) {
                auto tmp = vclt_u8(src.val[0], src.val[1]);
                auto mask = vreinterpretq_u16_u8(vcombine_u8(tmp, tmp));
                intermediate_result = vbslq_u16(mask,
                                               vmull_u8(delta,
                                                        vdup_n_u8(6)),
                                                        vdupq_n_u16(0));
            } else {
                intermediate_result = vmull_u8(delta, vdup_n_u8(mult));
            }
            intermediate_result = vaddw_u8(intermediate_result, sub);
            intermediate_result = vmulq_u16(intermediate_result, vdupq_n_u16(30));
            intermediate_result /= vreinterpretq_u16_u8(vcombine_u8(vzip_u8(delta, zero).val[0], vzip_u8(delta, zero).val[1]));
            
            result.val[0] = vbsl_u8(delta_positive_mask,
                                    vbsl_u8(color_mask,
                                            vuzp_u8(vget_low_u8(vreinterpretq_u8_u16(intermediate_result)), vget_high_u8(vreinterpretq_u8_u16(intermediate_result))).val[0],
                                            result.val[0]),
                                    zero);
            
            intermediate_result = vreinterpretq_u16_u8(vcombine_u8(vzip_u8(zero, delta).val[0], vzip_u8(zero, delta).val[1]));
            intermediate_result = vsubw_u8(intermediate_result, delta);
            intermediate_result /= vreinterpretq_u16_u8(vcombine_u8(vzip_u8(max, zero).val[0], vzip_u8(max, zero).val[1]));
            
            result.val[1] = vbsl_u8(color_mask, vbsl_u8(vtst_u8(max, zero),
                                                        zero,
                                                        vuzp_u8(vget_low_u8(vreinterpretq_u8_u16(intermediate_result)), vget_high_u8(vreinterpretq_u8_u16(intermediate_result))).val[0]),
                                    result.val[1]);
            result.val[2] = vbsl_u8(color_mask, max, result.val[2]);
        }
        
        vst3_u8(hsv, result);
    }

    auto t2_neon = chrono::high_resolution_clock::now();
    auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
    cout << "inside function rgb_to_gray_neon" << endl;
    cout << duration_neon << " us" << endl;
}


int main(int argc,char** argv)
{
    uint8_t * rgb_arr;
    uint8_t * hsv_arr;

    if (argc != 2) {
        cout << "Usage: opencv_neon image_name" << endl;
        return -1;
    }
    
    Mat rgb_image;
    rgb_image = imread(argv[1], IMREAD_COLOR);
    if (!rgb_image.data) {
        cout << "Could not open the image" << endl;
        return -1;
    }
    if (rgb_image.isContinuous()) {
        rgb_arr = rgb_image.data;
    }
    else {
        cout << "data is not continuous" << endl;
        return -2;
    }

    int width = rgb_image.cols;
    int height = rgb_image.rows;
    int num_pixels = width*height;
    Mat hsv_image(height, width, CV_8UC3, Scalar(0));
    hsv_arr = hsv_image.data;


    auto t1_neon = chrono::high_resolution_clock::now();
    rgb_to_hsv(rgb_arr, hsv_arr, num_pixels);
    auto t2_neon = chrono::high_resolution_clock::now();
    cvtColor(hsv_image, rgb_image, COLOR_HSV2BGR);
    auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
    cout << "rgb_to_gray_neon" << endl;
    cout << duration_neon << " us" << endl;

    imwrite("test.png", rgb_image);

    return 0;
}