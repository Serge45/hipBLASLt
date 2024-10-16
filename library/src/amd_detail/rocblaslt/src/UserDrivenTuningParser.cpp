#include "UserDrivenTuningParser.hpp"


#include <fstream>
#include <sstream>
#include <utility>

Tensile::DataType roc2TensileType(rocblaslt_compute_type type)
{
    switch(type)
    {
    case rocblaslt_compute_f32:
    case rocblaslt_compute_f32_fast_xf32:
    case rocblaslt_compute_f32_fast_f16:
    case rocblaslt_compute_f32_fast_bf16:
    case rocblaslt_compute_f32_fast_f8_fnuz:
    case rocblaslt_compute_f32_fast_bf8_fnuz:
    case rocblaslt_compute_f32_fast_f8bf8_fnuz:
    case rocblaslt_compute_f32_fast_bf8f8_fnuz:
#ifdef ROCM_USE_FLOAT8
    case rocblaslt_compute_f32_fast_f8_ocp:
    case rocblaslt_compute_f32_fast_bf8_ocp:
    case rocblaslt_compute_f32_fast_f8bf8_ocp:
    case rocblaslt_compute_f32_fast_bf8f8_ocp:
#endif
        return Tensile::DataType::Float;
    case rocblaslt_compute_f64:
        return Tensile::DataType::Double;
    case rocblaslt_compute_i32:
        return Tensile::DataType::Int32;
    case rocblaslt_compute_f16:
        return Tensile::DataType::Half;
    default:
        throw std::runtime_error("Unsupported type.");
    }
    return Tensile::DataType::None;
}

namespace Tensile
{

    std::unordered_map<ProblemOverride, int>
        getContractionProblemsFromFile(const std::string& path)
    {
        
        static std::unordered_map<ProblemOverride, int> m_override;
        
        if (m_override.size() == 0){
            
            std::ifstream file(path);
            std::string   line, entry;

            const auto delim         = ',';
            const int  max_entries   = 10;

            while(std::getline(file, line))
            {
                // Ignore lines without delimiter
                if(line.find(delim) == std::string::npos)
                {
                    continue;
                }

                std::vector<std::string> entries{};
                entries.reserve(max_entries);

                std::stringstream line_ss(line);
                while(getline(line_ss, entry, delim))
                {
                    entries.push_back(entry);
                }

                auto problemSolution = problemFromEntries(entries);
                if(problemSolution.second > 0)
                {
                    m_override.insert(problemSolution);
                }
            }
        }
        
        
        return m_override;
    }
    
    
    std::pair<ProblemOverride, int>
        problemFromEntries(const std::vector<std::string>& entries)
    {
        
        const size_t entries_n = entries.size();
        if(entries_n != 10)
        {
            return std::make_pair(ProblemOverride{}, -1);
        }

        //Expected: transA,transB,batch_count,M,N,K,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,input_type,output_type,compute_type,solution_index

        bool transA = (entries[0] != "N");
        bool transB = (entries[1] != "N");

        size_t m, n, b, k;
        DataType inputType   = DataType::None;
        DataType outputType  = DataType::None;
        DataType computeType = DataType::None;

        int solution_idx = -1;

        try
        {
            
            // To do:
            b = std::stol(entries[2]);
            m = std::stol(entries[3]);
            n = std::stol(entries[4]);
            k = std::stol(entries[5]);
            inputType   = hipDataType_to_tensile_type(string_to_hip_datatype(entries[6]));
            outputType  = hipDataType_to_tensile_type(string_to_hip_datatype(entries[7]));
            computeType = hipDataType_to_tensile_type(string_to_hip_datatype(entries[8]));
            solution_idx = std::stoi(entries[9]);

        }
        catch(std::invalid_argument const& ex)
        {
            return std::make_pair(ProblemOverride{}, -1);
        }
        catch(std::out_of_range const& ex)
        {
            return std::make_pair(ProblemOverride{}, -1);
        }

        if(inputType == DataType::None || outputType == DataType::None
           || computeType == DataType::None)
        {
            return std::make_pair(ProblemOverride{}, -1);
        }

        ProblemOverride  po(transA,
                            transB,
                            inputType,
                            computeType,
                            outputType,
                            m,
                            n,
                            k,
                            b);

        return std::make_pair(po, solution_idx);
        
    }

    
    ProblemOverride::ProblemOverride()
        : m_transA(false)
        , m_transB(false)
        , m_inputType(DataType::None)
        , m_computeType(DataType::None)
        , m_outputType(DataType::None)
        , m_m(0)
        , m_n(0)
        , m_k(0)
        , m_batchSize(0)
    {
    }

    
    ProblemOverride::ProblemOverride(bool     transA,
                                     bool     transB,
                                     DataType inputType,
                                     DataType computeType,
                                     DataType outputType,
                                     size_t   m,
                                     size_t   n,
                                     size_t   k,
                                     size_t   batchSize)
        : m_transA(transA)
        , m_transB(transB)
        , m_inputType(inputType)
        , m_computeType(computeType)
        , m_outputType(outputType)
        , m_m(m)
        , m_n(n)
        , m_k(k)
        , m_batchSize(batchSize)
    {
    }

    ProblemOverride::ProblemOverride(const RocblasltContractionProblem& problem)
    {
        if (problem.trans_a == HIPBLAS_OP_N)
            m_transA = false;
        else   
            m_transA = true;

        if (problem.trans_b == HIPBLAS_OP_N)
            m_transB = false;
        else   
            m_transB = true;
        m_inputType     = hipDataType_to_tensile_type(problem.a_type);
        m_computeType   = roc2TensileType(problem.compute_type);
        m_outputType    = hipDataType_to_tensile_type(problem.c_type);
        m_m             = problem.m;
        m_n             = problem.n;
        m_k             = problem.k;
        m_batchSize     = problem.batch_count;
        
    }

    
};
