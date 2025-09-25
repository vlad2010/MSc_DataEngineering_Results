#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

// Pass device_id as user_data for thread safety
void CL_CALLBACK buildCallback(cl_program program, void* user_data) {
    cl_int errcode;
    cl_build_status build_status;
    size_t build_log_size;
    cl_device_id device_id;

    // Extract device_id from user_data
    if (user_data == NULL) {
        fprintf(stderr, "Error: user_data (device_id) is NULL\n");
        return;
    }
    device_id = *(cl_device_id*)user_data;

    // Get build status
    errcode = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
    if (errcode != CL_SUCCESS) {
        fprintf(stderr, "Error getting build status: %d\n", errcode);
        return;
    }

    if (build_status == CL_BUILD_ERROR) {
        errcode = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
        if (errcode != CL_SUCCESS) {
            fprintf(stderr, "Error getting build log size: %d\n", errcode);
            return;
        }

        // Check for reasonable log size (arbitrary upper limit, e.g., 10MB)
        if (build_log_size == 0 || build_log_size > 10 * 1024 * 1024) {
            fprintf(stderr, "Build log size is invalid: %zu\n", build_log_size);
            return;
        }

        // Allocate space for log (+1 for null terminator)
        char* build_log = (char*)malloc(build_log_size + 1);
        if (build_log == NULL) {
            fprintf(stderr, "Memory allocation failed for build log\n");
            return;
        }

        errcode = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, NULL);
        if (errcode != CL_SUCCESS) {
            fprintf(stderr, "Error getting build log: %d\n", errcode);
            free(build_log);
            return;
        }

        // Ensure null-termination
        build_log[build_log_size] = '\0';

        // Print build log safely
        printf("Registro de compilación:\n%s\n", build_log);

        free(build_log);
    }
}