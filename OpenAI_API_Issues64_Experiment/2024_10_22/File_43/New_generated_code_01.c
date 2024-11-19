#include <stdio.h>
#include <CL/cl.h> // Include the OpenCL header

// Assume get_error_message function is defined somewhere
// const char* get_error_message(cl_int error_code);

// A safe maximum length for error messages
#define MAX_ERROR_MESSAGE_LENGTH 256

cl_int error_code = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
if (error_code != CL_SUCCESS) {
    const char* error_message = get_error_message(error_code);

    if (error_message != NULL) {
        // Truncate the message if it's too long, or ensure the messages are safe beforehand
        char safe_error_message[MAX_ERROR_MESSAGE_LENGTH];
        
        // Copy with a safe size check
        strncpy(safe_error_message, error_message, MAX_ERROR_MESSAGE_LENGTH - 1);
        safe_error_message[MAX_ERROR_MESSAGE_LENGTH - 1] = '\0'; // Ensure null-termination

        // Print the safe, truncated error message
        printf("Error: %s\n", safe_error_message);
    } else {
        // Handle the null error message case
        printf("Error: Unknown error occurred (code: %d)\n", error_code);
    }
}