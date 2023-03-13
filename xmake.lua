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

