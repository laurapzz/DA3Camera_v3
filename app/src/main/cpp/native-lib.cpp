#include <jni.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>
#include <vector>
#include <cmath>

// Standard TuSimple Row Anchors normalized (0.40 to 0.95)
const std::vector<float> ROW_ANCHORS = {
    0.40f, 0.41f, 0.42f, 0.43f, 0.44f, 0.45f, 0.46f, 0.47f, 0.48f, 0.49f,
    0.50f, 0.51f, 0.52f, 0.53f, 0.54f, 0.55f, 0.56f, 0.57f, 0.58f, 0.59f,
    0.60f, 0.61f, 0.62f, 0.63f, 0.64f, 0.65f, 0.66f, 0.67f, 0.68f, 0.69f,
    0.70f, 0.71f, 0.72f, 0.73f, 0.74f, 0.75f, 0.76f, 0.77f, 0.78f, 0.79f,
    0.80f, 0.81f, 0.82f, 0.83f, 0.84f, 0.85f, 0.86f, 0.87f, 0.88f, 0.89f,
    0.90f, 0.91f, 0.92f, 0.93f, 0.94f, 0.95f
};

cv::Mat generateBikeLaneROIMask(
    const float* loc_row,   // Pointer to Output 0 [1x100x56x4]
    const float* exist_row, // Pointer to Output 2 [1x2x56x4]
    int img_w,
    int img_h
) {
    std::vector<cv::Point> left_lane;  // Index 1 (Ego-Left)
    std::vector<cv::Point> right_lane; // Index 2 (Ego-Right)

    for (int lane_idx : {1, 2}) {
        for (int k = 0; k < 56; ++k) {
            float logit_0 = exist_row[0 * 56 * 4 + k * 4 + lane_idx];
            float logit_1 = exist_row[1 * 56 * 4 + k * 4 + lane_idx];
            float prob_exist = std::exp(logit_1) / (std::exp(logit_0) + std::exp(logit_1));

            if (prob_exist > 0.5f) {
                int max_grid = 0;
                float max_val = -1e9f;
                for (int g = 0; g < 100; ++g) {
                    float val = loc_row[g * 56 * 4 + k * 4 + lane_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_grid = g;
                    }
                }

                int x = static_cast<int>((max_grid / 99.0f) * img_w);
                int y = static_cast<int>(ROW_ANCHORS[k] * img_h);

                if (lane_idx == 1) left_lane.push_back(cv::Point(x, y));
                else right_lane.push_back(cv::Point(x, y));
            }
        }
    }

    cv::Mat mask = cv::Mat::zeros(img_h, img_w, CV_8UC1);
    if (left_lane.size() < 2 || right_lane.size() < 2) {
        return mask;
    }

    cv::Mat line_l, line_r;
    cv::fitLine(left_lane, line_l, cv::DIST_L2, 0, 0.01, 0.01);
    cv::fitLine(right_lane, line_r, cv::DIST_L2, 0, 0.01, 0.01);

    auto get_x = [](const cv::Mat& line, float y) -> int {
        float vx = line.at<float>(0);
        float vy = line.at<float>(1);
        float x0 = line.at<float>(2);
        float y0 = line.at<float>(3);
        if (std::abs(vy) < 1e-5) return static_cast<int>(x0);
        return static_cast<int>((vx / vy) * (y - y0) + x0);
    };

    cv::Point left_bottom(get_x(line_l, img_h), img_h);
    cv::Point right_bottom(get_x(line_r, img_h), img_h);

    std::vector<cv::Point> polygon;
    polygon.push_back(left_bottom);

    for (const auto& pt : left_lane) {
        polygon.push_back(pt);
    }

    for (auto it = right_lane.rbegin(); it != right_lane.rend(); ++it) {
        polygon.push_back(*it);
    }

    polygon.push_back(right_bottom);

    std::vector<std::vector<cv::Point>> pts_seq = { polygon };
    cv::fillPoly(mask, pts_seq, cv::Scalar(255));

    return mask;
}

extern "C"
JNIEXPORT jbyteArray JNICALL
Java_com_example_da3camera_Ufldv2Utils_nativeGenerateBikeLaneROIMask(
    JNIEnv* env,
    jobject thiz,
    jfloatArray locRow_,
    jfloatArray existRow_,
    jint imgW,
    jint imgH
) {
    jfloat* loc_row = env->GetFloatArrayElements(locRow_, nullptr);
    jfloat* exist_row = env->GetFloatArrayElements(existRow_, nullptr);

    cv::Mat mask = generateBikeLaneROIMask(loc_row, exist_row, imgW, imgH);

    env->ReleaseFloatArrayElements(locRow_, loc_row, JNI_ABORT);
    env->ReleaseFloatArrayElements(existRow_, exist_row, JNI_ABORT);

    int total = mask.total();
    jbyteArray byteArray = env->NewByteArray(total);
    env->SetByteArrayRegion(byteArray, 0, total, reinterpret_cast<jbyte*>(mask.data));
    return byteArray;
}
