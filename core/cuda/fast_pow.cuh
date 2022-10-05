template <int ip = -1, bool positive = false> __device__ float pow_fun(float x, float p) {
    float t1, t2;
    switch (ip) {
        case 0: return 1.0f;
        case 1: return positive ? x : abs(x);
        case 2: return x * x;
        case 3: return (positive ? x : abs(x)) * x * x;
        case 4: t1 = x * x; return t1 * t1;
        case 5: t1 = x * x; return (positive ? x : abs(x)) * t1 * t1;
        case 6: t1 = x * x; return t1 * t1 * t1;
        case 7: t1 = x * x; return (positive ? x : abs(x)) * t1 * t1 * t1;
        case 8: t1 = x * x; t1 = t1 * t1; return t1 * t1;
        case 9: t1 = x * x; t1 = t1 * t1; return (positive ? x : abs(x)) * t1 * t1;
        case 10: t1 = x * x; t2 = t1 * t1; return t2 * t2 * t1;
        case 11: t1 = x * x; t2 = t1 * t1; return (positive ? x : abs(x)) * t2 * t2 * t1;
        case 12: t1 = x * x * x; return t1 * t1 * t1;
        case 13: t1 = x * x * x; return (positive ? x : abs(x)) * t1 * t1 * t1;
        case 14: t1 = x * x; t2 = t1 * t1; return t2 * t2 * t2 * t1;
        case 15: t1 = x * x; t2 = t1 * t1 * (positive ? x : abs(x)); return t2 * t2 * t2;
        case 16: t1 = x * x; t1 = t1 * t1; t1 = t1 * t1; return t1 * t1;
        default: return positive ? __powf(x, p) : __powf(abs(x), p);
    }
}