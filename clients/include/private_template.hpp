//marco記得大寫
#define use_host_vector(host_input, paramType, operations, ...)\
    if (paramType==HIP_R_32F)       template_cast<host_vector<float>*>(host_input, [&](auto& vec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if (paramType==HIP_R_64F)  template_cast<host_vector<double>*>(host_input, [&](auto& vec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if (paramType==HIP_R_16BF) template_cast<host_vector<hip_bfloat16>*>(host_input, [&](auto& vec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if (paramType==HIP_R_8F_E4M3_FNUZ) template_cast<host_vector<hipblaslt_f8_fnuz>*>(host_input, [&](auto& vec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if (paramType==HIP_R_8F_E5M2_FNUZ) template_cast<host_vector<hipblaslt_bf8_fnuz>*>(host_input, [&](auto& vec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if (paramType==HIP_R_32I)  template_cast<host_vector<int32_t>*>(host_input, [&](auto& vec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if (paramType==HIP_R_8I)   template_cast<host_vector<hipblasLtInt8>*>(host_input, [&](auto& vec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else template_cast<host_vector<hipblasLtHalf>*>(host_input, [&](auto& vec, auto&&... args) { operations ; }, __VA_ARGS__) ;

#define use_host_vector_Ti(host_input, bias_data, paramType, enable_bias, hBias_index, i, Ti)\
    if (paramType==HIP_R_32F)       template_cast_Ti<host_vector<float>*, Ti>(host_input, bias_data, enable_bias, hBias_index, i) ; \
    else if (paramType==HIP_R_64F)  template_cast_Ti<host_vector<double>*, Ti>(host_input, bias_data, enable_bias, hBias_index, i) ; \
    else if (paramType==HIP_R_16BF) template_cast_Ti<host_vector<hip_bfloat16>*, Ti>(host_input, bias_data, enable_bias, hBias_index, i) ; \
    else if (paramType==HIP_R_8F_E4M3_FNUZ) template_cast_Ti<host_vector<hipblaslt_f8_fnuz>*, Ti>(host_input, bias_data, enable_bias, hBias_index, i) ; \
    else if (paramType==HIP_R_8F_E5M2_FNUZ) template_cast_Ti<host_vector<hipblaslt_bf8_fnuz>*, Ti>(host_input, bias_data, enable_bias, hBias_index, i) ; \
    else if (paramType==HIP_R_32I)  template_cast_Ti<host_vector<int32_t>*, Ti>(host_input, bias_data, enable_bias, hBias_index, i) ; \
    else if (paramType==HIP_R_8I)   template_cast_Ti<host_vector<hipblasLtInt8>*, Ti>(host_input, bias_data, enable_bias, hBias_index, i) ; \
    else template_cast_Ti<host_vector<hipblasLtHalf>*, Ti>(host_input, bias_data, enable_bias, hBias_index, i) ;

#define use_device_vector(device_input, paramType, operations, ...)\
    if (paramType==HIP_R_32F)       template_cast<device_vector<float>*>(device_input, [&](auto& vec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if (paramType==HIP_R_64F)  template_cast<device_vector<double>*>(device_input, [&](auto& vec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if (paramType==HIP_R_16BF) template_cast<device_vector<hip_bfloat16>*>(device_input, [&](auto& vec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if (paramType==HIP_R_8F_E4M3_FNUZ) template_cast<device_vector<hipblaslt_f8_fnuz>*>(device_input, [&](auto& vec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if (paramType==HIP_R_8F_E5M2_FNUZ) template_cast<device_vector<hipblaslt_bf8_fnuz>*>(device_input, [&](auto& vec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if (paramType==HIP_R_32I)  template_cast<device_vector<int32_t>*>(device_input, [&](auto& vec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if (paramType==HIP_R_8I)   template_cast<device_vector<hipblasLtInt8>*>(device_input, [&](auto& vec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else template_cast<device_vector<hipblasLtHalf>*>(device_input, [&](auto& vec, auto&&... args) { operations ; }, __VA_ARGS__) ;

#define use_device_host_vector(device_input, host_input, paramType, operations, ...)\
    if(paramType==HIP_R_32F)        template_cast2<device_vector<float>*, host_vector<float>*>(device_input, host_input, [&](auto& dvec, auto& hvec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if(paramType==HIP_R_64F)   template_cast2<device_vector<double>*, host_vector<double>*>(device_input, host_input, [&](auto& dvec, auto& hvec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if(paramType==HIP_R_16BF)  template_cast2<device_vector<hip_bfloat16>*, host_vector<hip_bfloat16>*>(device_input, host_input, [&](auto& dvec, auto& hvec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if(paramType==HIP_R_8F_E4M3_FNUZ)template_cast2<device_vector<hipblaslt_f8_fnuz>*, host_vector<hipblaslt_f8_fnuz>*>(device_input, host_input, [&](auto& dvec, auto& hvec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if(paramType==HIP_R_8F_E5M2_FNUZ)template_cast2<device_vector<hipblaslt_bf8_fnuz>*, host_vector<hipblaslt_bf8_fnuz>*>(device_input, host_input, [&](auto& dvec, auto& hvec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if(paramType==HIP_R_32I)   template_cast2<device_vector<int32_t>*, host_vector<int32_t>*>(device_input, host_input, [&](auto& dvec, auto& hvec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if(paramType==HIP_R_8I)    template_cast2<device_vector<hipblasLtInt8>*, host_vector<hipblasLtInt8>*>(device_input, host_input, [&](auto& dvec, auto& hvec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else template_cast2<device_vector<hipblasLtHalf>*, host_vector<hipblasLtHalf>*>(device_input, host_input, [&](auto& dvec, auto& hvec, auto&&... args) { operations ; }, __VA_ARGS__) ;

#define use_device_host_gold_vector(device_input, host_input, gold_input, paramType, operations, ...)\
    if(paramType==HIP_R_32F)        template_cast3<device_vector<float>*, host_vector<float>*>(device_input, host_input, gold_input, [&](auto& dvec, auto& hvec, auto& gvec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if(paramType==HIP_R_64F)   template_cast3<device_vector<double>*, host_vector<double>*>(device_input, host_input, gold_input, [&](auto& dvec, auto& hvec, auto& gvec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if(paramType==HIP_R_16BF)  template_cast3<device_vector<hip_bfloat16>*, host_vector<hip_bfloat16>*>(device_input, host_input, gold_input, [&](auto& dvec, auto& hvec, auto& gvec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if(paramType==HIP_R_8F_E4M3_FNUZ)template_cast3<device_vector<hipblaslt_f8_fnuz>*, host_vector<hipblaslt_f8_fnuz>*>(device_input, host_input, gold_input, [&](auto& dvec, auto& hvec, auto& gvec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if(paramType==HIP_R_8F_E5M2_FNUZ)template_cast3<device_vector<hipblaslt_bf8_fnuz>*, host_vector<hipblaslt_bf8_fnuz>*>(device_input, host_input, gold_input, [&](auto& dvec, auto& hvec, auto& gvec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if(paramType==HIP_R_32I)   template_cast3<device_vector<int32_t>*, host_vector<int32_t>*>(device_input, host_input, gold_input, [&](auto& dvec, auto& hvec, auto& gvec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else if(paramType==HIP_R_8I)    template_cast3<device_vector<hipblasLtInt8>*, host_vector<hipblasLtInt8>*>(device_input, host_input, gold_input, [&](auto& dvec, auto& hvec, auto& gvec, auto&&... args) { operations ; }, __VA_ARGS__) ; \
    else template_cast3<device_vector<hipblasLtHalf>*, host_vector<hipblasLtHalf>*>(device_input, host_input, gold_input, [&](auto& dvec, auto& hvec, auto& gvec, auto&&... args) { operations ; }, __VA_ARGS__) ;

#define use_host_vector_hipblaslt_init(h_input, i, paramType, in1, in2, in3)\
    if (paramType==HIP_R_32F) hipblaslt_init<float>(*template_cast_return<std::vector<host_vector<float>*>>(h_input)[i], in1, in2, in3); \
    else if (paramType==HIP_R_64F) hipblaslt_init<double>(*template_cast_return<std::vector<host_vector<double>*>>(h_input)[i], in1, in2, in3); \
    else if (paramType==HIP_R_16BF) hipblaslt_init<hip_bfloat16>(*template_cast_return<std::vector<host_vector<hip_bfloat16>*>>(h_input)[i], in1, in2, in3); \
    else if (paramType==HIP_R_8F_E4M3_FNUZ) hipblaslt_init<hipblaslt_f8_fnuz>(*template_cast_return<std::vector<host_vector<hipblaslt_f8_fnuz>*>>(h_input)[i], in1, in2, in3); \
    else if (paramType==HIP_R_8F_E5M2_FNUZ) hipblaslt_init<hipblaslt_bf8_fnuz>(*template_cast_return<std::vector<host_vector<hipblaslt_bf8_fnuz>*>>(h_input)[i], in1, in2, in3); \
    else if (paramType==HIP_R_32I) hipblaslt_init<int32_t>(*template_cast_return<std::vector<host_vector<int32_t>*>>(h_input)[i], in1, in2, in3); \
    else if (paramType==HIP_R_8I) hipblaslt_init<hipblasLtInt8>(*template_cast_return<std::vector<host_vector<hipblasLtInt8>*>>(h_input)[i], in1, in2, in3); \
    else hipblaslt_init<hipblasLtHalf>(*template_cast_return<std::vector<host_vector<hipblasLtHalf>*>>(h_input)[i], in1, in2, in3);


#define template_cast_host(h_input, paramType)\
    ((paramType==HIP_R_32F) ? template_cast_return<std::vector<host_vector<float>*>>(h_input) \
    : (paramType==HIP_R_64F) ? template_cast_return<std::vector<host_vector<double>*>>(h_input) \
    : (paramType==HIP_R_16F) ? template_cast_return<std::vector<host_vector<hipblasLtHalf>*>>(h_input) \
    : (paramType==HIP_R_16BF) ? template_cast_return<std::vector<host_vector<hip_bfloat16>*>>(h_input) \
    : (paramType==HIP_R_8F_E4M3_FNUZ) ? template_cast_return<std::vector<host_vector<hipblaslt_f8_fnuz>*>>(h_input) \
    : (paramType==HIP_R_8F_E5M2_FNUZ) ? template_cast_return<std::vector<host_vector<hipblaslt_bf8_fnuz>*>>(h_input) \
    : (paramType==HIP_R_32I) ? template_cast_return<std::vector<host_vector<int32_t>*>>(h_input) \
    : (paramType==HIP_R_8I) ? template_cast_return<std::vector<host_vector<hipblasLtInt8>*>>(h_input) \
    : std::vector<host_vector<signed char>*>() )

#define template_cast_device(d_input, paramType)\
    ((paramType==HIP_R_32F) ? template_cast_return<std::vector<device_vector<float>*>>(d_input) \
    : (paramType==HIP_R_64F) ? template_cast_return<std::vector<device_vector<double>*>>(d_input) \
    : (paramType==HIP_R_16F) ? template_cast_return<std::vector<device_vector<hipblasLtHalf>*>>(d_input) \
    : (paramType==HIP_R_16BF) ? template_cast_return<std::vector<device_vector<hip_bfloat16>*>>(d_input) \
    : (paramType==HIP_R_8F_E4M3_FNUZ) ? template_cast_return<std::vector<device_vector<hipblaslt_f8_fnuz>*>>(d_input) \
    : (paramType==HIP_R_8F_E5M2_FNUZ) ? template_cast_return<std::vector<device_vector<hipblaslt_bf8_fnuz>*>>(d_input) \
    : (paramType==HIP_R_32I) ? template_cast_return<std::vector<device_vector<int32_t>*>>(d_input) \
    : (paramType==HIP_R_8I) ? template_cast_return<std::vector<device_vector<hipblasLtInt8>*>>(d_input) \
    : std::vector<device_vector<signed char>*>() )


#define new_device_vector(vec_ptr, i_input, paramType, param_1, param_2, param_3)\
    if(paramType==HIP_R_32F)\
        static_cast<std::vector<device_vector<float>*>*>(vec_ptr)->at(i_input) = new device_vector<float>(param_1, param_2, param_3);\
    else if(paramType==HIP_R_64F)\
        static_cast<std::vector<device_vector<double>*>*>(vec_ptr)->at(i_input) = new device_vector<double>(param_1, param_2, param_3);\
    else if(paramType==HIP_R_16F)\
        static_cast<std::vector<device_vector<hipblasLtHalf>*>*>(vec_ptr)->at(i_input) = new device_vector<hipblasLtHalf>(param_1, param_2, param_3);\
    else if(paramType==HIP_R_16BF)\
        static_cast<std::vector<device_vector<hip_bfloat16>*>*>(vec_ptr)->at(i_input) = new device_vector<hip_bfloat16>(param_1, param_2, param_3);\
    else if(paramType==HIP_R_8F_E4M3_FNUZ)\
        static_cast<std::vector<device_vector<hipblaslt_f8_fnuz>*>*>(vec_ptr)->at(i_input) = new device_vector<hipblaslt_f8_fnuz>(param_1, param_2, param_3);\
    else if(paramType==HIP_R_8F_E5M2_FNUZ)\
        static_cast<std::vector<device_vector<hipblaslt_bf8_fnuz>*>*>(vec_ptr)->at(i_input) = new device_vector<hipblaslt_bf8_fnuz>(param_1, param_2, param_3);\
    else if(paramType==HIP_R_32I)\
        static_cast<std::vector<device_vector<int32_t>*>*>(vec_ptr)->at(i_input) = new device_vector<int32_t>(param_1, param_2, param_3);\
    else if(paramType==HIP_R_8I)\
        static_cast<std::vector<device_vector<hipblasLtInt8>*>*>(vec_ptr)->at(i_input) = new device_vector<hipblasLtInt8>(param_1, param_2, param_3);


#define delete_device_vector(vec_ptr, i_input, paramType)\
    if(paramType==HIP_R_32F)\
        delete_device_vector_type(static_cast<std::vector<device_vector<float>*>*>(vec_ptr)->at(i_input));\
    else if(paramType==HIP_R_64F)\
        delete_device_vector_type(static_cast<std::vector<device_vector<double>*>*>(vec_ptr)->at(i_input));\
    else if(paramType==HIP_R_16F)\
        delete_device_vector_type(static_cast<std::vector<device_vector<hipblasLtHalf>*>*>(vec_ptr)->at(i_input));\
    else if(paramType==HIP_R_16BF)\
        delete_device_vector_type(static_cast<std::vector<device_vector<hip_bfloat16>*>*>(vec_ptr)->at(i_input));\
    else if(paramType==HIP_R_8F_E4M3_FNUZ)\
        delete_device_vector_type(static_cast<std::vector<device_vector<hipblaslt_f8_fnuz>*>*>(vec_ptr)->at(i_input));\
    else if(paramType==HIP_R_8F_E5M2_FNUZ)\
        delete_device_vector_type(static_cast<std::vector<device_vector<hipblaslt_bf8_fnuz>*>*>(vec_ptr)->at(i_input));\
    else if(paramType==HIP_R_32I)\
        delete_device_vector_type(static_cast<std::vector<device_vector<int32_t>*>*>(vec_ptr)->at(i_input));\
    else if(paramType==HIP_R_8I)\
        delete_device_vector_type(static_cast<std::vector<device_vector<hipblasLtInt8>*>*>(vec_ptr)->at(i_input));


#define new_host_vector(vec_ptr, i_input, paramType, param_1)\
    if(paramType==HIP_R_32F)\
        static_cast<std::vector<host_vector<float>*>*>(vec_ptr)->at(i_input) = new host_vector<float>(param_1);\
    else if(paramType==HIP_R_64F)\
        static_cast<std::vector<host_vector<double>*>*>(vec_ptr)->at(i_input) = new host_vector<double>(param_1);\
    else if(paramType==HIP_R_16F)\
        static_cast<std::vector<host_vector<hipblasLtHalf>*>*>(vec_ptr)->at(i_input) = new host_vector<hipblasLtHalf>(param_1);\
    else if(paramType==HIP_R_16BF)\
        static_cast<std::vector<host_vector<hip_bfloat16>*>*>(vec_ptr)->at(i_input) = new host_vector<hip_bfloat16>(param_1);\
    else if(paramType==HIP_R_8F_E4M3_FNUZ)\
        static_cast<std::vector<host_vector<hipblaslt_f8_fnuz>*>*>(vec_ptr)->at(i_input) = new host_vector<hipblaslt_f8_fnuz>(param_1);\
    else if(paramType==HIP_R_8F_E5M2_FNUZ)\
        static_cast<std::vector<host_vector<hipblaslt_bf8_fnuz>*>*>(vec_ptr)->at(i_input) = new host_vector<hipblaslt_bf8_fnuz>(param_1);\
    else if(paramType==HIP_R_32I)\
        static_cast<std::vector<host_vector<int32_t>*>*>(vec_ptr)->at(i_input) = new host_vector<int32_t>(param_1);\
    else if(paramType==HIP_R_8I)\
        static_cast<std::vector<host_vector<hipblasLtInt8>*>*>(vec_ptr)->at(i_input) = new host_vector<hipblasLtInt8>(param_1);


#define type_to_param(Type)\
    if(std::is_same<Type, float>::value) paramType_Ti=HIP_R_32F;  \
    else if(std::is_same<Type, double>::value) paramType_Ti=HIP_R_64F;  \
    else if(std::is_same<Type, hip_bfloat16>::value) paramType_Ti=HIP_R_16BF; \
    else if(std::is_same<Type, hipblaslt_f8_fnuz>::value) paramType_Ti=HIP_R_8F_E4M3_FNUZ; \
    else if(std::is_same<Type, hipblaslt_bf8_fnuz>::value) paramType_Ti=HIP_R_8F_E5M2_FNUZ; \
    else if(std::is_same<Type, int32_t>::value) paramType_Ti=HIP_R_32I; \
    else if(std::is_same<Type, hipblasLtInt8>::value) paramType_Ti=HIP_R_8I; \
    else  paramType_Ti=HIP_R_16F;

#define param_to_type(paramType)\
    if(paramType_Ti==HIP_R_32F)  using Type = float  ;  \
    else if(paramType_Ti==HIP_R_64F) using Type = double ;  \
    else if(paramType_Ti==HIP_R_16BF) using Type = hip_bfloat16;  \
    else if(paramType_Ti==HIP_R_8F_E4M3_FNUZ) using Type = hipblaslt_f8_fnuz; \
    else if(paramType_Ti==HIP_R_8F_E5M2_FNUZ)using Type = hipblaslt_bf8_fnuz ; \
    else if(paramType_Ti==HIP_R_32I) using Type = int32_t   ; \
    else if(paramType_Ti==HIP_R_8I)  using Type = hipblasLtInt8   ; \
    else using Type = hipblasLtHalf  ;
