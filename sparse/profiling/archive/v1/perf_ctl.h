// Gate `perf stat` / `perf record` counting to a code region, so index
// construction does not pollute search-phase counters.
//
// Run under:
//   mkfifo ctl.fifo ack.fifo
//   PERF_CTL_FIFO=ctl.fifo PERF_ACK_FIFO=ack.fifo \
//     perf stat -D -1 --control=fifo:ctl.fifo,ack.fifo -e <events> -- <binary> <args>
//
// (-D -1 starts with counters disabled; enable()/disable() toggle them.)
// If the env vars are unset every call is a no-op, so unprofiled runs are
// completely unaffected. profiling/perf_groups.sh sets all of this up.
#pragma once
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>

namespace perf_ctl {

inline int& ctl_fd() { static int fd = -1; return fd; }
inline int& ack_fd() { static int fd = -1; return fd; }

inline void init() {
    const char* c = getenv("PERF_CTL_FIFO");
    const char* a = getenv("PERF_ACK_FIFO");
    if (!c || !a) return;
    ctl_fd() = open(c, O_WRONLY);
    ack_fd() = open(a, O_RDONLY);
}

inline void send(const char* cmd) {
    if (ctl_fd() < 0) return;
    if (write(ctl_fd(), cmd, strlen(cmd)) < 0) return;
    char buf[16];
    ssize_t n = read(ack_fd(), buf, sizeof(buf));  // wait for perf's "ack\n"
    (void)n;
}

inline void enable()  { send("enable\n"); }
inline void disable() { send("disable\n"); }

} // namespace perf_ctl
