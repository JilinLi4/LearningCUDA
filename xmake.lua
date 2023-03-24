add_rules("mode.debug", "mode.release")

add_cugencodes("native")
add_cugencodes("compute_75")

target("HelloWorldCUDA")
    set_kind("binary")
    add_files("src/main.cu")

target("Add")
    -- add_defines("USE_DP")
    set_kind("binary")
    add_files("src/add.cu")

target("memory")
    set_kind("binary")
    add_files("src/memery.cu")

target("stream")
    set_kind("binary")
    add_files("src/stream.cu")

target("reduce_sum")
    set_kind("binary")
    add_files("src/reduce_sum.cu")

target("reduce_sum1")
    set_kind("binary")
    add_files("src/reduce_sum_du.cu")

