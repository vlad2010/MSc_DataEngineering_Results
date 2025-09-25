#include <stdio.h>
#include <CL/cl.h>
#include <assert.h>

// Assuming this function exists in your codebase
const char* get_error_message(cl_int error_code);

// Fixed version with security improvements
cl_int execute_kernel_safely(cl_command_queue command_queue, 
                             cl_kernel kernel,
                             const size_t* global_work_size,
                             const size_t* local_work_size) {
    
    // CWE-20 Fix: Validate input parameters
    if (command_queue == NULL) {
        fprintf(stderr, "Error: Invalid command queue (NULL)\n");
        return CL_INVALID_COMMAND_QUEUE;
    }
    
    if (kernel == NULL) {
        fprintf(stderr, "Error: Invalid kernel (NULL)\n");
        return CL_INVALID_KERNEL;
    }
    
    if (global_work_size == NULL) {
        fprintf(stderr, "Error: Invalid global work size (NULL)\n");
        return CL_INVALID_GLOBAL_WORK_SIZE;
    }
    
    // Validate work sizes are non-zero (assuming 2D kernel)
    for (int i = 0; i < 2; i++) {
        if (global_work_size[i] == 0) {
            fprintf(stderr, "Error: Global work size dimension %d is zero\n", i);
            return CL_INVALID_GLOBAL_WORK_SIZE;
        }
        
        // If local_work_size is provided, validate it
        if (local_work_size != NULL && local_work_size[i] == 0) {
            fprintf(stderr, "Error: Local work size dimension %d is zero\n", i);
            return CL_INVALID_WORK_GROUP_SIZE;
        }
    }
    
    // Execute the kernel
    cl_int error_code = clEnqueueNDRangeKernel(
        command_queue, 
        kernel, 
        2,                  // 2D kernel
        NULL,              // global work offset
        global_work_size, 
        local_work_size,   // can be NULL for automatic determination
        0,                 // num events in wait list
        NULL,              // event wait list
        NULL               // event
    );
    
    if (error_code != CL_SUCCESS) {
        const char* error_message = get_error_message(error_code);
        
        // CWE-252 Fix: Check for NULL return value
        if (error_message == NULL) {
            fprintf(stderr, "Error: Unknown error code %d\n", error_code);
        } else {
            // CWE-134 Fix: Use format string properly
            // Never use external string as format string directly
            fprintf(stderr, "Error: %s (code: %d)\n", error_message, error_code);
            // Alternative safe approaches:
            // fputs("Error: ", stderr);
            // fputs(error_message, stderr);
            // fprintf(stderr, "\n");
        }
    }
    
    return error_code;
}

// Example usage
int main() {
    // Assuming these are properly initialized elsewhere
    cl_command_queue command_queue = /* initialized queue */;
    cl_kernel kernel = /* initialized kernel */;
    
    size_t global_work_size[2] = {1024, 1024};
    size_t local_work_size[2] = {16, 16};
    
    cl_int result = execute_kernel_safely(
        command_queue, 
        kernel, 
        global_work_size, 
        local_work_size
    );
    
    if (result == CL_SUCCESS) {
        printf("Kernel executed successfully\n");
    }
    
    return result == CL_SUCCESS ? 0 : 1;
}