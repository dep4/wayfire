decoration = shared_module('decoration',
    ['decoration.cpp', 'deco-subsurface.cpp', 'deco-button.cpp',
      'deco-layout.cpp', 'deco-theme.cpp'],
    include_directories: [wayfire_api_inc, wayfire_conf_inc, plugins_common_inc],
    dependencies: [wlroots, pixman, wf_protos, wfconfig, cairo, pango, pangocairo, plugins_pch_pthread_dep],
    cpp_args: plugins_pch_pthread_args,
    install: true,
    install_dir: join_paths(get_option('libdir'), 'wayfire'))
