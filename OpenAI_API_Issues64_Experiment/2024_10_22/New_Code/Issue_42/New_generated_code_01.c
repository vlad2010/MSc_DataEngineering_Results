#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

void CL_CALLBACK buildCallback(cl_program program, void* user_data) {
    cl_int errcode;
    cl_build_status build_status;
    size_t build_log_size;
    cl_device_id device_id = *((cl_device_id *)user_data); // Assuming user_data contains a valid device ID

    // Obtain the program's build status
    errcode = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
    if (errcode != CL_SUCCESS) {
        fprintf(stderr, "Error obtaining program build status: %d\n", errcode);
        return;
    }

    // If the build status is CL_BUILD_ERROR, obtain the build log
    if (build_status == CL_BUILD_ERROR) {
        errcode = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
        if (errcode != CL_SUCCESS) {
            fprintf(stderr, "Error obtaining build log size: %d\n", errcode);
            return;
        }

        if (build_log_size > 0) {
            char* build_log = (char*)malloc(build_log_size);
            if (build_log == NULL) {
                fprintf(stderr, "Memory allocation for build log failed\n");
                return;
            }

            errcode = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, NULL);
            if (errcode != CL_SUCCESS) {
                fprintf(stderr, "Error obtaining build log: %d\n", errcode);
                free(build_log);
                return;
            }

            // Print build log
            printf("Build log:\n%s\n", build_log);

            free(build_log);
        } else {
            fprintf(stderr, "Build log size is zero\n");
        }
    }
}