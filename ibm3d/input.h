#ifndef INPUT_H
#define INPUT_H

typedef struct _input_params InputParams;

InputParams *InputParams_new(void);
void InputParams_read_file(InputParams *, const char *);
void InputParams_destroy(InputParams *);

#endif
