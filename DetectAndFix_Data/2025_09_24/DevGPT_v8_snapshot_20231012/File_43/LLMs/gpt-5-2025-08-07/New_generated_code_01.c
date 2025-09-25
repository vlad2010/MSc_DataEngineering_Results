#include <CL/cl.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

/* Safe mapping of OpenCL error codes to constant strings.
   Always returns a non-NULL pointer. */
static const char* cl_error_to_string(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "Success";
        case CL_DEVICE_NOT_FOUND: return "Device not found";
        case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE: return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES: return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling info not available";
        case CL_MEM_COPY_OVERLAP: return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH: return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE: return "Program build failure";
        case CL_MAP_FAILURE: return "Map failure";
#ifdef CL_MISALIGNED_SUB_BUFFER_OFFSET
        case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "Misaligned sub-buffer offset";
#endif
#ifdef CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "Exec status error for events in wait list";
#endif
#ifdef CL_COMPILE_PROGRAM_FAILURE
        case CL_COMPILE_PROGRAM_FAILURE: return "Program compile failure";
#endif
#ifdef CL_LINKER_NOT_AVAILABLE
        case CL_LINKER_NOT_AVAILABLE: return "Linker not available";
#endif
#ifdef CL_LINK_PROGRAM_FAILURE
        case CL_LINK_PROGRAM_FAILURE: return "Program link failure";
#endif
#ifdef CL_DEVICE_PARTITION_FAILED
        case CL_DEVICE_PARTITION_FAILED: return "Device partition failed";
#endif
#ifdef CL_KERNEL_ARG_INFO_NOT_AVAILABLE
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: return "Kernel arg info not available";
#endif
        case CL_INVALID_VALUE: return "Invalid value";
        case CL_INVALID_DEVICE_TYPE: return "Invalid device type";
        case CL_INVALID_PLATFORM: return "Invalid platform";
        case CL_INVALID_DEVICE: return "Invalid device";
        case CL_INVALID_CONTEXT: return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES: return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE: return "Invalid command queue";
        case CL_INVALID_HOST_PTR: return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT: return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE: return "Invalid image size";
        case CL_INVALID_SAMPLER: return "Invalid sampler";
        case CL_INVALID_BINARY: return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS: return "Invalid build options";
        case CL_INVALID_PROGRAM: return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME: return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION: return "Invalid kernel definition";
        case CL_INVALID_KERNEL: return "Invalid kernel";
        case CL_INVALID_ARG_INDEX: return "Invalid argument index";
        case CL_INVALID_ARG_VALUE: return "Invalid argument value";
        case CL_INVALID_ARG_SIZE: return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS: return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION: return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE: return "Invalid work-group size";
        case CL_INVALID_WORK_ITEM_SIZE: return "Invalid work-item size";
        case CL_INVALID_GLOBAL_OFFSET: return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST: return "Invalid event wait list";
        case CL_INVALID_EVENT: return "Invalid event";
        case CL_INVALID_OPERATION: return "Invalid operation";
        case CL_INVALID_GL_OBJECT: return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE: return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL: return "Invalid mip level";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "Invalid global work size";
        default: return "Unknown OpenCL error";
    }
}

static void log_opencl_error(const char* api, cl_int err) {
    const char* msg = cl_error_to_string(err);
    /* CWE-209: log to stderr; avoid leaking excessive info in production builds */
    (void)fprintf(stderr, "%s failed (err=%d): %s\n", api, (int)err, msg);
    (void)fflush(stderr);
}

/* Basic validation to prevent out-of-bounds and common invalid sizes.
   This mitigates CWE-125/787 by ensuring arrays have at least work_dim elements. */
static bool validate_ndrange(cl_uint work_dim,
                             const size_t* global_work_size, size_t global_work_size_len,
                             const size_t* local_work_size,  size_t local_work_size_len) {
    if (work_dim == 0 || work_dim > 3) {
        return false; /* CL only allows 1..3 dimensions */
    }
    if (!global_work_size || global_work_size_len < work_dim) {
        return false;
    }
    for (cl_uint i = 0; i < work_dim; ++i) {
        if (global_work_size[i] == 0) {
            return false; /* zero-sized global range is invalid */
        }
    }
    if (local_work_size) {
        if (local_work_size_len < work_dim) {
            return false;
        }
        for (cl_uint i = 0; i < work_dim; ++i) {
            if (local_work_size[i] == 0) {
                return false;
            }
            /* Basic divisibility check to avoid CL_INVALID_WORK_GROUP_SIZE at runtime */
            if ((global_work_size[i] % local_work_size[i]) != 0) {
                return false;
            }
        }
    }
    return true;
}

/* Safe wrapper that validates input and logs errors correctly. */
cl_int enqueue_nd_range_kernel_safe(cl_command_queue command_queue,
                                    cl_kernel kernel,
                                    cl_uint work_dim,
                                    const size_t* global_work_size, size_t global_work_size_len,
                                    const size_t* local_work_size,  size_t local_work_size_len,
                                    cl_event* event_out) {
    if (kernel == NULL || command_queue == NULL) {
        log_opencl_error("clEnqueueNDRangeKernel", CL_INVALID_VALUE);
        return CL_INVALID_VALUE;
    }

    if (!validate_ndrange(work_dim, global_work_size, global_work_size_len,
                          local_work_size, local_work_size_len)) {
        log_opencl_error("clEnqueueNDRangeKernel (validation)", CL_INVALID_VALUE);
        return CL_INVALID_VALUE;
    }

    cl_int err = clEnqueueNDRangeKernel(command_queue, kernel,
                                        work_dim,
                                        /* global_work_offset */ NULL,
                                        global_work_size,
                                        local_work_size,
                                        /* num_events_in_wait_list */ 0,
                                        /* event_wait_list */ NULL,
                                        event_out);
    if (err != CL_SUCCESS) {
        log_opencl_error("clEnqueueNDRangeKernel", err);
        return err;
    }

    return CL_SUCCESS;
}

/* Example usage:
   Ensure you set kernel args, create command_queue, etc., before calling. */
void example_usage(cl_command_queue queue, cl_kernel kernel) {
    size_t gws[2] = {1024, 1024};
    size_t lws[2] = {16, 16};
    cl_event evt = NULL;

    cl_int err = enqueue_nd_range_kernel_safe(queue, kernel,
                                              /* work_dim */ 2,
                                              gws, 2,
                                              lws, 2,
                                              &evt);
    if (err != CL_SUCCESS) {
        /* CWE-703: Stop or clean up on error; do not continue as if it succeeded. */
        return;
    }

    /* Wait for completion if you need results before proceeding */
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        log_opencl_error("clFinish", err);
    }

    if (evt) {
        clReleaseEvent(evt);
    }
}