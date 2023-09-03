tile = shared_module('simple-tile',
        ['tile-plugin.cpp', 'tree.cpp', 'tree-controller.cpp'],
        include_directories: [wayfire_api_inc, wayfire_conf_inc, plugins_common_inc, grid_inc, wobbly_inc],
        cpp_args: plugins_pch_args,
        dependencies: [wlroots, pixman, wfconfig, plugins_pch_dep],
        install: true,
        install_dir: join_paths(get_option('libdir'), 'wayfire'))
