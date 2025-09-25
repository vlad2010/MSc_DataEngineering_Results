cl_int error_code = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
if (error_code != CL_SUCCESS) {
    const char* error_message = get_error_message(error_code);

    // Defensive: handle NULL pointer from get_error_message
    if (error_message == NULL) {
        fprintf(stderr, "Error: Unknown error occurred (code: %d)\n", error_code);
    } else {
        // Defensive: use a fixed format string to avoid format string vulnerabilities
        fprintf(stderr, "Error: %s (code: %d)\n", error_message, error_code);
    }

    // Optionally: handle error (cleanup, exit, etc.)
}