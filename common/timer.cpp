#include "timer.hpp"
#include <string>

HRTimer::HRTimer() {
    str_func_name = "";
    str_logfile_name = "";
    start();
}

HRTimer::HRTimer(std::string func_name) {
    str_func_name = func_name;
    str_logfile_name = "";
    start();
}

HRTimer::HRTimer(std::string func_name, std::string logfile_name) {
    str_func_name = func_name;
    str_logfile_name = logfile_name;
    start();
}

HRTimer::~HRTimer() {
    if (m_bDumpedTime == false) {
        stop();
        //printtime_ms();
    }
}

void HRTimer::start() {
    start_t = high_resolution_clock::now();
    m_bDumpedTime = false;
}

void HRTimer::stop() {
    end_t = high_resolution_clock::now();
    gettime_ms();
}

std::size_t HRTimer::gettime_ms() {
    auto dur = end_t - start_t;
    milli_sec = duration_cast<milliseconds>(dur).count();
    return milli_sec;
}

std::size_t HRTimer::gettime_us() {
    auto dur = end_t - start_t;
    auto usec = duration_cast<microseconds>(dur).count();
    return usec;
}

void HRTimer::printtime_us() {
    auto usec = gettime_us();
    std::cout << usec;
}


void HRTimer::dumptime_stream(std::ostream &os) {
    if (str_func_name.empty() == false) {
        os << "Function " << str_func_name << " exec time:" << "\t";
    }
    os << milli_sec << " ms" << std::endl;
    m_bDumpedTime = true;
}

void HRTimer::printtime_ms() {
    if (str_logfile_name.empty() == false) {
        std::ofstream ofs(str_logfile_name, std::ios::app);
        if (ofs.is_open() == true) {
            dumptime_stream(ofs);
        }
        else {
            dumptime_stream(std::cout);
        }
    }
    else {
        dumptime_stream(std::cout);
    }
}

