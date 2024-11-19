cl_int errcode;
cl_program program;

// Crea y compila tu programa de OpenCL
program = clCreateProgramWithSource(context, 1, &source, NULL, &errcode);
// Configura la función de control de errores
errcode = clSetProgramBuildCallback(program, NULL, buildCallback, NULL);
errcode = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
