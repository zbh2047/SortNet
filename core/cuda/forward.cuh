#include <utility>
#include <tuple>

template <typename Tuple, std::size_t... Is>
auto pop_front_impl(const Tuple& tuple, std::index_sequence<Is...>) {
    return std::make_tuple(std::get<1 + Is>(tuple)...);
}

template <typename Tuple> struct Call {
    template<template<int...> typename Function, bool... F, int... T, typename... Args>
    static auto call(std::integer_sequence<int, T...> unused, Tuple tuple, Args... args) {
        bool first = std::get<0>(tuple);
        auto other = pop_front_impl(tuple, std::make_index_sequence<std::tuple_size<Tuple>::value - 1>());
        if (first) return Call<decltype(other)>::template call<Function, F..., true>(unused, other, args...);
        else return Call<decltype(other)>::template call<Function, F..., false>(unused, other, args...);
    }
};

template <> struct Call<std::tuple<>> {
    template<template<int...> typename Function, bool... F, int... T, typename... Args>
    static auto call(std::integer_sequence<int, T...> unused, std::tuple<> tuple, Args... args) {
        return Function<T..., F...>::call(args...);
    }
};

#define create_helper(name, prefix) \
template <int... CONSTS> \
struct name##_helper { \
    template <typename... Args> \
    static void call(dim3 dimGrid, dim3 dimBlock, Args... args) { \
        prefix##_##name##_kernel<CONSTS...><<<dimGrid, dimBlock>>>(args...); \
    } \
};
