#ifndef CL_GLOBALS_H
#define CL_GLOBALS_H

cl_device_id device;
cl_context context;
cl_program program;
cl_command_queue queue;

cl_mem error_buffer;
cl_mem x_buffer;
cl_mem y_buffer;
cl_mem z_buffer;
cl_mem dt_buffer;
cl_mem vx_buffer;
cl_mem vy_buffer;
cl_mem vz_buffer;
cl_mem ax_buffer;
cl_mem ay_buffer;
cl_mem az_buffer;
cl_mem t_buffer;
cl_mem rad_buffer;
cl_mem mass_buffer;
cl_mem start_buffer;
cl_mem sort_buffer;
cl_mem count_buffer;
cl_mem children_buffer;
cl_mem collisions_buffer;
cl_mem bottom_node_buffer;
cl_mem softening2_buffer;
cl_mem inv_opening_angle2_buffer;
cl_mem minimum_collision_velocity_buffer;
cl_mem maxdepth_buffer;
cl_mem boxsize_buffer;
cl_mem rootx_buffer;
cl_mem rooty_buffer;
cl_mem rootz_buffer;
cl_mem collisions_max2_r_buffer;
cl_mem num_nodes_buffer;
cl_mem num_bodies_buffer;
cl_mem OMEGA_buffer;
cl_mem OMEGAZ_buffer;
cl_mem sindt_buffer;
cl_mem tandt_buffer;
cl_mem sindtz_buffer;
cl_mem tandtz_buffer;
cl_mem G_buffer;

cl_int error;
cl_int work_groups;
cl_int *children_host;
cl_int *collisions_host;
cl_float *x_host;
cl_float *y_host;
cl_float *z_host;
cl_float *cellcenter_x_host;
cl_float *cellcenter_y_host;
cl_float *cellcenter_z_host;
cl_float *vx_host;
cl_float *vy_host;
cl_float *vz_host;
cl_float *ax_host;
cl_float *ay_host;
cl_float *az_host;
cl_float *mass_host;
cl_int *count_host;
cl_int *sort_host;
cl_int *start_host;
cl_float *rad_host;
cl_int nghostx;
cl_int nghosty;
cl_int nghostz;
cl_float t_host;
cl_float softening2_host;
cl_float inv_opening_angle2_host;
cl_float minimum_collision_velocity_host;
cl_float collisions_max_r_host;
cl_float collisions_max2_r_host;
cl_int bottom_node_host;
cl_int maxdepth_host;
cl_int num_nodes_host;
cl_int num_bodies_host;
cl_float boxsize_host;
cl_float rootx_host;
cl_float rooty_host;			      
cl_float rootz_host;
cl_float OMEGA_host;
cl_float OMEGAZ_host;
cl_float sindt_host;
cl_float tandt_host;
cl_float sindtz_host;
cl_float tandtz_host;
cl_float dt_host;
cl_float G_host;
cl_float surface_density;
cl_float particle_density;
cl_float particle_radius_min;
cl_float particle_radius_max;
cl_float particle_radius_slope;

cl_kernel tree_kernel;
cl_kernel tree_kernel_no_mass;
cl_kernel tree_gravity_kernel;
cl_kernel tree_collisions_kernel;
cl_kernel tree_sort_kernel;
cl_kernel force_gravity_kernel;
cl_kernel collisions_search_kernel;
cl_kernel collisions_resolve_kernel;
cl_kernel boundaries_kernel;
cl_kernel integrator_part1_kernel;
cl_kernel integrator_part2_kernel;

size_t local_size_tree_kernel;
size_t global_size_tree_kernel;
size_t local_size_tree_gravity_kernel;
size_t global_size_tree_gravity_kernel;
size_t local_size_tree_sort_kernel;
size_t global_size_tree_sort_kernel;
size_t local_size_force_gravity_kernel;
size_t wave_fronts_in_force_gravity_kernel;
size_t global_size_force_gravity_kernel;
size_t local_size_collisions_search_kernel;
size_t wave_fronts_in_collisions_search_kernel;
size_t global_size_collisions_search_kernel;
size_t local_size_collisions_resolve_kernel;
size_t global_size_collisions_resolve_kernel;
size_t local_size_boundaries_kernel;
size_t global_size_boundaries_kernel;
size_t local_size_integrator_kernel;
size_t global_size_integrator_kernel;


#endif


