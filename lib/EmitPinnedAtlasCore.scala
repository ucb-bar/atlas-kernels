package atlas.tile

import atlas.common.AtlasParams
import freechips.rocketchip.tilelink.TLBundleParameters

object EmitPinnedAtlasCore extends App {
  val host = TLBundleParameters(addressBits = 64, dataBits = 32, sourceBits = 1,
    sinkBits = 1, sizeBits = 7, echoFields = Nil, requestFields = Nil,
    responseFields = Nil, hasBCE = false)
  val data = TLBundleParameters(addressBits = 64, dataBits = 256, sourceBits = 6,
    sinkBits = 1, sizeBits = 7, echoFields = Nil, requestFields = Nil,
    responseFields = Nil, hasBCE = false)
  circt.stage.ChiselStage.emitSystemVerilogFile(
    new AtlasCore(AtlasParams(), host, host, data, data),
    Array("--target-dir", args(0), "--split-verilog"),
    Array("--disable-all-randomization"))
}
