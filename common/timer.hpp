#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include <iostream>
#include <ctime>
#include <chrono>
#include <string>
#include <fstream>

using namespace std::chrono;

class HRTimer {
public:
    HRTimer();
    HRTimer(std::string func_name);
    HRTimer(std::string func_name, std::string logfile_name);
    ~HRTimer();

    void start();
    void stop();

    size_t gettime_ms();
    size_t gettime_us();
    void printtime_ms();
    void printtime_us();

private:
    void dumptime_stream(std::ostream &os);

private:
    std::string str_func_name;
    std::string str_logfile_name;

    high_resolution_clock::time_point start_t, end_t;
    size_t milli_sec;
    bool m_bDumpedTime;
};

#endif

