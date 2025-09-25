cl_int error_code = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
if (error_code != CL_SUCCESS) {
    const char* error_message = get_error_message(error_code);
    printf("Error: %s\n
