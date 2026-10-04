void max_pool_3x3_s2_p1(const int8_t *input, int8_t *output, int ih, int iw, int ic, int oh, int ow) {
    for (int c = 0; c < ic; c++) {
        for (int h = 0; h < oh; h++) {
            for (int w = 0; w < ow; w++) {
                int8_t max_val = -128;
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int ih_idx = h * 2 - 1 + kh;
                        int iw_idx = w * 2 - 1 + kw;
                        if (ih_idx >= 0 && ih_idx < ih && iw_idx >= 0 && iw_idx < iw) {
                            int8_t val = input[(ih_idx * iw + iw_idx) * ic + c];
                            if (val > max_val) max_val = val;
                        } else {
                            if (0 > max_val) max_val = 0; // Padding is usually 0
                        }
                    }
                }
                output[(h * ow + w) * ic + c] = max_val;
            }
        }
    }
}
