// Drive the selected AtlasCore TileLink ports without a Chipyard/CPU harness.
// This replay deliberately does not claim full EE290 system qualification.
#include "VAtlasCore.h"
#include "VAtlasCore___024root.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <deque>
#include <map>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include "dma_config.h"
#include <sys/resource.h>

class Replay {
  VerilatedContext context;
  VAtlasCore dut{&context};
  VerilatedVcdC trace;
  uint64_t time = 0, cycles = 0, limit;
  bool tracing=false, checkDma=false;
  std::string memoryMode;
  std::ofstream dmaLog;
  double previousReady=0;
  std::deque<bool> recentStores;
  uint64_t firstIssue=0, ecallCycle=0, observed=0;
  struct Response { uint64_t due; unsigned source; std::array<uint32_t,8> data; uint64_t address, accepted; unsigned opcode; };
  std::deque<Response> pending;
  std::map<uint64_t,uint8_t> dram;
  unsigned reads=0,writes=0,astalls=0,dwaits=0;
  unsigned resetCount=0,startCount=0,imemLoadCount=0,vmemInitPasses=0;
  std::vector<std::pair<uint64_t,uint64_t>> mutableDram;
  void service() {
    dut.io_dmaTL_d_valid = !pending.empty() && pending.front().due <= cycles;
    dut.io_dmaTL_a_ready = memoryMode=="synthetic" ? cycles%11 < 4 :
      (pending.size()<atlas_dma_queue_depth || (dut.io_dmaTL_d_valid && dut.io_dmaTL_d_ready));
    if(dut.io_dmaTL_d_valid) {
      dut.io_dmaTL_d_bits_source=pending.front().source;
      for(unsigned i=0;i<8;++i) dut.io_dmaTL_d_bits_data[i]=pending.front().data[i];
    }
  }
  void accept() {
    if(dut.io_dmaTL_a_valid && !dut.io_dmaTL_a_ready) ++astalls;
    if(!pending.empty() && !dut.io_dmaTL_d_valid) ++dwaits;
    bool response=dut.io_dmaTL_d_valid && dut.io_dmaTL_d_ready;
    if(response) {
      const auto &r=pending.front();
      logDma("response",r);
      pending.pop_front();
    }
    if(dut.io_dmaTL_a_valid && dut.io_dmaTL_a_ready) {
      auto addr=uint64_t(dut.io_dmaTL_a_bits_address);
      auto op=unsigned(dut.io_dmaTL_a_bits_opcode);
      if((addr&31) || (op!=0 && op!=4)) throw std::runtime_error("unsupported DMA request");
      uint64_t due=cycles+43;
      if(memoryMode=="npu-model") {
        recentStores.push_back(op!=4);
        if(recentStores.size()>atlas_dma_window_beats) recentStores.pop_front();
        unsigned stores=std::count(recentStores.begin(),recentStores.end(),true);
        double readPct=100.0*(recentStores.size()-stores)/recentStores.size();
        auto selected=atlas_default_dma_curves[0];
        // Curves are sorted: strict comparison preserves the lower tie.
        for(auto row:atlas_default_dma_curves)
          if(std::abs(row.read_pct-readPct)<std::abs(selected.read_pct-readPct)) selected=row;
        previousReady=std::max(double(cycles)+selected.latency,previousReady+1.0/selected.peak);
        due=uint64_t(std::ceil(previousReady-1e-9));
      }
      Response r{due,unsigned(dut.io_dmaTL_a_bits_source),{},addr,cycles,op};
      logDma("request",r);
      for(unsigned i=0;i<32;++i) {
        if(!dram.count(addr+i)) throw std::runtime_error("DMA escaped initialized memory");
        if(op==4) r.data[i/4]|=uint32_t(dram.at(addr+i))<<(8*(i%4));
        else dram[addr+i]=(dut.io_dmaTL_a_bits_data[i/4]>>(8*(i%4)))&255;
      }
      if(op==4) ++reads; else ++writes;
      pending.push_back(r);

    }

  }
  void logDma(const char *event,const Response &r) {
    if(dmaLog) dmaLog<<event<<','<<cycles<<','<<r.source<<','<<r.address<<','<<r.opcode<<','<<r.due<<','<<r.accepted<<'\n';
  }
  void evaluate() {
    dut.eval(); if(tracing) trace.dump(time); ++time;
    if (context.gotFinish()) throw std::runtime_error("RTL stopped or asserted");
  }
  void exchange(CData &valid, const CData &ready, const CData &response, const char *port) {
    auto wait = [&](const CData &signal, const char *phase) {
      unsigned waits = 0;
      while (!signal) {
        if (++waits > 100) throw std::runtime_error(std::string(port) + phase + " handshake timeout");
        tick();
      }
    };
    valid = 1; dut.clock = 0; evaluate(); wait(ready, " A");
    tick(); valid = 0; evaluate(); wait(response, " D");
  }
public:
  Replay(const char *wave, uint64_t cycleLimit, const std::string &mode, const char *dmaPath) : limit(cycleLimit), memoryMode(mode) {
    if(memoryMode!="synthetic" && memoryMode!="npu-model") throw std::runtime_error("invalid memory mode");
    if(dmaPath) {
      dmaLog.open(dmaPath);
      if(!dmaLog) throw std::runtime_error("cannot open DMA trace");
      dmaLog<<"event,absolute_cycle,source,address,opcode,due_cycle,accepted_cycle\n";
    }
    if(wave && *wave) { tracing=true; context.traceEverOn(true); dut.trace(&trace, 3); trace.open(wave); }
    dut.clock = 0; dut.reset = 1;
    dut.io_imemTL_a_valid = dut.io_csrTL_a_valid = dut.io_vmemTL_a_valid = 0;
    dut.io_imemTL_d_ready = dut.io_csrTL_d_ready = dut.io_vmemTL_d_ready = 1;
    dut.io_dmaTL_a_ready = 1;
    dut.io_dmaTL_d_valid = 0; dut.io_dmaTL_d_bits_source = 0;
    for (unsigned i = 0; i < 8; ++i) dut.io_dmaTL_d_bits_data[i] = 0;
    evaluate();
    for (unsigned i = 0; i < 4; ++i) tick();
    dut.reset = 0; tick();
    ++resetCount;
  }
  ~Replay() { dut.final(); trace.close(); }
  void tick() {
    if (++cycles > limit) throw std::runtime_error("cycle bound exceeded");
    dut.clock = 0; service(); evaluate(); // inputs settle before the rising edge
    if(!dut.reset) {
      if(dut.rootp->AtlasCore__DOT__scalar__DOT__s1_fire) { if(!firstIssue) firstIssue=cycles; ++observed; }
      if(dut.rootp->AtlasCore__DOT__csrfile__DOT__io_csr_set_ecall && !ecallCycle) { ecallCycle=cycles; ++observed; if(!firstIssue) firstIssue=cycles; }
    }
    accept(); dut.clock = 1; evaluate(); dut.clock = 0; evaluate();
  }
  uint32_t scalar(bool csr, bool write, uint32_t address, uint32_t data = 0) {
    (csr ? dut.io_csrTL_a_bits_opcode : dut.io_imemTL_a_bits_opcode) = write ? 0 : 4;
    (csr ? dut.io_csrTL_a_bits_size : dut.io_imemTL_a_bits_size) = 2;
    (csr ? dut.io_csrTL_a_bits_source : dut.io_imemTL_a_bits_source) = 0;
    (csr ? dut.io_csrTL_a_bits_address : dut.io_imemTL_a_bits_address) = address;
    (csr ? dut.io_csrTL_a_bits_data : dut.io_imemTL_a_bits_data) = data;
    exchange(csr ? dut.io_csrTL_a_valid : dut.io_imemTL_a_valid, csr ? dut.io_csrTL_a_ready : dut.io_imemTL_a_ready,
             csr ? dut.io_csrTL_d_valid : dut.io_imemTL_d_valid, "scalar");
    uint32_t result = csr ? dut.io_csrTL_d_bits_data : dut.io_imemTL_d_bits_data;
    auto opcode = csr ? dut.io_csrTL_d_bits_opcode : dut.io_imemTL_d_bits_opcode;
    if (opcode != (write ? 0 : 1)) throw std::runtime_error("scalar response opcode mismatch");
    tick(); return result;
  }
  std::array<uint32_t, 8> memory(bool write, unsigned line, const std::array<uint32_t, 8> &data = {}) {
    dut.io_vmemTL_a_bits_opcode = write ? 0 : 4;
    dut.io_vmemTL_a_bits_size = 5; // 32 bytes: one full physical line
    dut.io_vmemTL_a_bits_source = 0; dut.io_vmemTL_a_bits_address = 0x20000000U + 32 * line;
    dut.io_vmemTL_a_bits_mask = 0xffffffffU;
    for (unsigned i = 0; i < 8; ++i) dut.io_vmemTL_a_bits_data[i] = data[i];
    exchange(dut.io_vmemTL_a_valid, dut.io_vmemTL_a_ready, dut.io_vmemTL_d_valid, "VMEM");
    std::array<uint32_t, 8> result;
    for (unsigned i = 0; i < 8; ++i) result[i] = dut.io_vmemTL_d_bits_data[i];
    if (dut.io_vmemTL_d_bits_opcode != (write ? 0 : 1)) throw std::runtime_error("VMEM response opcode mismatch");
    tick(); return result;
  }
  // Memory contents come from the case image; every initialized byte is checked.
  struct Region { uint64_t base; std::vector<uint8_t> bytes; };
  std::vector<Region> vmemInit, dramInit, vmemExpect, dramExpect;
  unsigned expectedReads = 0, expectedWrites = 0;
  static std::vector<uint8_t> hex(const std::string &text) {
    if (text.size() % 2) throw std::runtime_error("odd image hex length");
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < text.size(); i += 2) bytes.push_back(std::stoul(text.substr(i, 2), nullptr, 16));
    return bytes;
  }
  void load(const char *path, bool allowMutable = false) {
    // New image views do not clear the retained device or off-chip memory.
    vmemInit.clear(); dramInit.clear(); vmemExpect.clear(); dramExpect.clear();
    expectedReads = expectedWrites = 0;
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read case image");
    std::string kind, data; uint64_t base;
    while (input >> kind) {
      if (kind == "dma") { input >> expectedReads >> expectedWrites; checkDma=true; continue; }
      if (kind == "mutable_dram") {
        uint64_t size;
        if (!allowMutable || !(input >> base >> size) || !size || base + size < base)
          throw std::runtime_error("invalid session mutable DRAM declaration");
        mutableDram.emplace_back(base, size); continue;
      }
      if (!(input >> base >> data)) throw std::runtime_error("malformed case image");
      Region region{base, hex(data)};
      bool vmem = kind == "vmem" || kind == "expect_vmem";
      if (vmem && (region.bytes.empty() || region.bytes.size() % 32)) throw std::runtime_error("VMEM region is not whole lines");
      if (kind == "vmem") vmemInit.push_back(region);
      else if (kind == "dram") dramInit.push_back(region);
      else if (kind == "expect_vmem") vmemExpect.push_back(region);
      else if (kind == "expect_dram") dramExpect.push_back(region);
      else throw std::runtime_error("unknown case image record");
    }
    if (!input.eof()) throw std::runtime_error("malformed case image");
  }
  void run(const std::vector<uint32_t> &program, const char *image, const char *output, uint64_t outputBase, uint64_t outputSize) {
    load(image);
    for (auto &r : dramInit) for (size_t i = 0; i < r.bytes.size(); ++i) dram[r.base + i] = r.bytes[i];
    scalar(true,true,0x40018,0);
    if(!(scalar(true,false,0x40008)&1)) throw std::runtime_error("initial halt missing");
    for(unsigned i=0;i<program.size();++i) scalar(false,true,0x20000+4*i,program[i]);
    for(unsigned i=0;i<program.size();++i) if(scalar(false,false,0x20000+4*i)!=program[i]) throw std::runtime_error("IMEM mismatch");
    for (auto &r : vmemInit)
      for (size_t line = 0; line < r.bytes.size() / 32; ++line) {
        std::array<uint32_t,8> d{};
        for(unsigned i=0;i<32;++i) d[i/4]|=uint32_t(r.bytes[32*line+i])<<(8*(i%4));
        memory(true,r.base+line,d);
      }
    for(uint32_t offset:{0x10U,0x14U,0U,4U}) scalar(true,true,0x40000+offset,0);
    scalar(true,true,0x40018,1);
    uint32_t status=0,marker=0;unsigned poll=0;
    for(;cycles+64<limit;++poll) { status=scalar(true,false,0x40008);marker=scalar(true,false,0x40010);if((status&1)&&((status>>1)&3)) break; }
    auto illegal=scalar(true,false,0x4000c);
    std::ofstream binary(output,std::ios::binary);
    if(!binary) throw std::runtime_error("cannot write actual output");
    for(uint64_t i=0;i<outputSize;++i) { auto it=dram.find(outputBase+i); if(it==dram.end()) throw std::runtime_error("output escaped mapped DRAM"); binary.put(char(it->second)); }
    binary.close();
    unsigned mismatches=0;
    if(status!=5||illegal||!pending.empty()||!firstIssue||!ecallCycle) ++mismatches;
    if(checkDma && (reads!=expectedReads||writes!=expectedWrites)) ++mismatches;
    size_t dramBytes = 0, vmemBytes = 0;
    for (auto &r : dramExpect)
      for (size_t i = 0; i < r.bytes.size(); ++i, ++dramBytes)
        if (!dram.count(r.base + i) || dram.at(r.base + i) != r.bytes[i]) {
          if(mismatches<10) std::cerr<<"DRAM byte mismatch address="<<r.base+i<<'\n'; ++mismatches;
        }
    for (auto &r : vmemExpect)
      for (size_t line = 0; line < r.bytes.size() / 32; ++line) {
        auto d=memory(false,r.base+line);
        for(unsigned i=0;i<32;++i,++vmemBytes) if(((d[i/4]>>(8*(i%4)))&255)!=r.bytes[32*line+i]) {
          if(mismatches<10) std::cerr<<"VMEM byte mismatch line="<<r.base+line<<" byte="<<i<<'\n'; ++mismatches;
        }
      }
    std::cout<<"ATLAS_KERNEL_RESULT execution_cycles="<<(ecallCycle>=firstIssue ? ecallCycle-firstIssue+1 : 0)<<" observed_instructions="<<observed
      <<" reads="<<reads<<" writes="<<writes<<" status="<<status<<" marker="<<marker<<" illegal_pc="<<illegal<<" mismatches="<<mismatches
      <<" memory="<<memoryMode<<" checked_dram_bytes="<<dramBytes<<" checked_vmem_bytes="<<vmemBytes<<" total_cycles="<<cycles<<" output_bytes="<<outputSize<<std::endl;
    if(mismatches) throw std::runtime_error("kernel completion/output mismatch");
  }
};

static void safePath(const char *name) {
  auto path=std::filesystem::absolute(name);
  for(const auto &part:path) {
    if(part=="..") throw std::runtime_error("parent traversal refused");
    auto value=part.string(); std::transform(value.begin(),value.end(),value.begin(),[](unsigned char c){return std::tolower(c);});
    if(value.find("ham" "mer")!=std::string::npos || value.find("vl" "si")!=std::string::npos) throw std::runtime_error("restricted path");
  }
  std::filesystem::path prefix;
  for(const auto &part:path) { prefix/=part; if(std::filesystem::is_symlink(std::filesystem::symlink_status(prefix))) throw std::runtime_error("symlink refused"); }
}
int main(int argc,char **argv) {
  const rlimit coreLimit{0,0}; setrlimit(RLIMIT_CORE,&coreLimit);
  if(argc<7) { std::cerr<<"program.hex max-cycles case.image output.bin output-address output-bytes [trace.vcd] [--memory synthetic|npu-model] [--dma-trace path] required\n"; return 2; }
  try {
    for(int index:{1,3,4}) safePath(argv[index]);
    std::string memory="synthetic";
    const char *wave=nullptr,*dmaPath=nullptr;
    for(int i=7;i<argc;++i) {
      std::string option=argv[i];
      if(option=="--memory" && i+1<argc) memory=argv[++i];
      else if(option=="--dma-trace" && i+1<argc) { dmaPath=argv[++i]; safePath(dmaPath); }
      else if(option.rfind("--",0)!=0 && !wave) { wave=argv[i]; safePath(wave); }
      else throw std::runtime_error("unknown or incomplete harness option");
    }
    std::ifstream input(argv[1]); if(!input) throw std::runtime_error("cannot read words");
    std::vector<uint32_t> words; uint64_t word;
    while(input>>std::hex>>word) { if(word>0xffffffffULL || words.size()>=32768) throw std::runtime_error("invalid word"); words.push_back(word); }
    if(!input.eof()||words.empty()) throw std::runtime_error("malformed words");
    Replay replay(wave,std::stoull(argv[2]),memory,dmaPath);
    replay.run(words,argv[3],argv[4],std::stoull(argv[5],nullptr,0),std::stoull(argv[6],nullptr,0));
    return 0;
  } catch(const std::exception &e) { std::cerr<<"ATLAS_KERNEL_FAILED "<<e.what()<<'\n'; return 1; }
}
