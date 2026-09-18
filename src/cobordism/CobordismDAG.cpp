// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/CobordismDAG.h"

#include <stdexcept>

#include "cobordism/MultiCobordism.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using complexd = std::complex<double>;

int CobordismDAG::addNode(std::shared_ptr<Spacetime> host,
                          const std::vector<std::vector<complexd>> &literalInputs,
                          const std::vector<std::pair<int, int>> &upstream,
                          const std::vector<std::vector<complexd>> &outputTargets,
                          const std::vector<int> &degrees, double gamma,
                          std::uint64_t seed) {
  nodes_.push_back(Node{std::move(host), literalInputs, upstream, outputTargets,
                        degrees, gamma, seed});
  return static_cast<int>(nodes_.size()) - 1;
}

void CobordismDAG::run(int stage1MaxSteps, int stage1CandidateMoves,
                       double stage2Beta, int stage2MaxIters) {
  const std::size_t n = nodes_.size();
  outputs_.assign(n, {});
  residuals_.assign(n, 0.0);
  done_.assign(n, false);
  outputFibers_.assign(n, {});
  fiberRefusals_.assign(n, "");
  pipedInputs_.assign(n, 0);
  twoBodyReads_.assign(n, std::nullopt);
  if (attachments_.size() < n) attachments_.resize(n);
  if (twoBodyTargets_.size() < n) twoBodyTargets_.resize(n);

  std::size_t completed = 0;
  while (completed < n) {
    bool progressed = false;
    for (std::size_t i = 0; i < n; ++i) {
      if (done_[i]) continue;
      bool ready = true;
      for (const auto &e : nodes_[i].upstream)
        if (e.first < 0 || e.first >= static_cast<int>(n) || !done_[e.first]) {
          ready = false;
          break;
        }
      if (!ready) continue;

      const Node &nd = nodes_[i];
      // Assemble input targets: literals, then each upstream node's chosen output.
      std::vector<std::vector<complexd>> inputs = nd.literalInputs;
      for (const auto &e : nd.upstream) {
        const auto &up = outputs_[e.first];
        if (e.second < 0 || e.second >= static_cast<int>(up.size()))
          throw std::out_of_range("CobordismDAG::run: bad upstream output index");
        inputs.push_back(up[e.second]);
      }

      MultiCobordism opt(nd.host, inputs, nd.outputTargets, nd.degrees,
                            nd.gamma, nd.seed);
      // One construct seed vertex per block (the first |inputs|+|outputs| ids).
      const auto verts = nd.host->getVertexList()->toVector();
      std::vector<std::uint64_t> inSeeds, outSeeds;
      std::size_t v = 0;
      for (; v < verts.size() && inSeeds.size() < inputs.size(); ++v)
        inSeeds.push_back(verts[v]->getId());
      for (; v < verts.size() && outSeeds.size() < nd.outputTargets.size(); ++v)
        outSeeds.push_back(verts[v]->getId());
      opt.seedInputs(inSeeds);
      opt.seedOutputs(outSeeds);
      if (pipeFibers_) {
        opt.useFiberResiduals(scoreBlocksByFiber_);
        // Pipe upstream output fibers into the downstream input blocks the
        // edges name (input slots after the literals), beside the targets.
        std::size_t slot = nd.literalInputs.size();
        for (const auto &e : nd.upstream) {
          const auto &fibers = outputFibers_[e.first];
          if (e.second < static_cast<int>(fibers.size()) && fibers[e.second] &&
              slot < opt.inputs().size()) {
            const auto attachment = attachments_[i].find(static_cast<int>(slot));
            if (attachment != attachments_[i].end())
              opt.attachInputFiber(slot, *fibers[e.second], attachment->second);
            else
              opt.setInputFiber(slot, *fibers[e.second]);
            ++pipedInputs_[i];
          }
          ++slot;
        }
        if (twoBodyTargets_[i]) opt.setTwoBodyTarget(twoBodyTargets_[i]->chi, twoBodyTargets_[i]->choiDecomposed);
      }
      opt.runStage1(stage1MaxSteps, stage1CandidateMoves);
      opt.runStage2(stage2Beta, stage2MaxIters);

      residuals_[i] = opt.rU(opt.spacetime());
      if (twoBodyTargets_[i]) {
        try {
          twoBodyReads_[i] = opt.readTwoBody();
        } catch (const std::exception &ex) {
          fiberRefusals_[i] = ex.what();
        }
      }
      outputs_[i] = nd.outputTargets;  // verified outputs, threaded downstream
      if (pipeFibers_) {
        outputFibers_[i].assign(opt.outputs().size(), std::nullopt);
        for (std::size_t j = 0; j < opt.outputs().size(); ++j) {
          try {
            outputFibers_[i][j] = opt.readOutputFiber(j, fiberDegree_);
          } catch (const std::exception &ex) {
            fiberRefusals_[i] = ex.what();
          }
        }
      }
      done_[i] = true;
      ++completed;
      progressed = true;
    }
    if (!progressed)
      throw std::runtime_error("CobordismDAG::run: cycle (no runnable node)");
  }
}

std::vector<complexd> CobordismDAG::output(int node, int outputIndex) const {
  if (node < 0 || node >= static_cast<int>(outputs_.size()))
    throw std::out_of_range("CobordismDAG::output: node id out of range");
  const auto &outs = outputs_[node];
  if (outputIndex < 0 || outputIndex >= static_cast<int>(outs.size()))
    throw std::out_of_range("CobordismDAG::output: output index out of range");
  return outs[outputIndex];
}

int CobordismDAG::numOutputs(int node) const {
  if (node < 0 || node >= static_cast<int>(outputs_.size()))
    throw std::out_of_range("CobordismDAG::numOutputs: node id out of range");
  return static_cast<int>(outputs_[node].size());
}

double CobordismDAG::residual(int node) const {
  if (node < 0 || node >= static_cast<int>(residuals_.size()))
    throw std::out_of_range("CobordismDAG::residual: node id out of range");
  return residuals_[node];
}



void CobordismDAG::setInputAttachment(int node, int slot, std::vector<std::vector<std::uint64_t>> cells) {
  if (node < 0 || node >= static_cast<int>(nodes_.size()))
    throw std::out_of_range("CobordismDAG::setInputAttachment: node index out of range");
  if (slot < 0) throw std::invalid_argument("CobordismDAG::setInputAttachment: negative slot");
  if (attachments_.size() < nodes_.size()) attachments_.resize(nodes_.size());
  attachments_[static_cast<std::size_t>(node)][slot] = std::move(cells);
}

void CobordismDAG::setTwoBodyTarget(int node, Eigen::MatrixXcd chi, bool choiDecomposed) {
  if (node < 0 || node >= static_cast<int>(nodes_.size()))
    throw std::out_of_range("CobordismDAG::setTwoBodyTarget: node index out of range");
  if (twoBodyTargets_.size() < nodes_.size()) twoBodyTargets_.resize(nodes_.size());
  twoBodyTargets_[static_cast<std::size_t>(node)] =
      MultiCobordism::TwoBodyTarget{std::move(chi), choiDecomposed};
}

const MultiCobordism::TwoBodyRead &CobordismDAG::twoBodyRead(int node) const {
  if (!hasTwoBodyRead(node))
    throw std::logic_error("CobordismDAG::twoBodyRead: the node carries no two-body reading");
  return *twoBodyReads_[static_cast<std::size_t>(node)];
}

bool CobordismDAG::hasTwoBodyRead(int node) const {
  return node >= 0 && node < static_cast<int>(twoBodyReads_.size()) &&
         twoBodyReads_[static_cast<std::size_t>(node)].has_value();
}

void CobordismDAG::setFiberPiping(bool enabled, int degree, bool scoreBlocksByFiber) {
  pipeFibers_ = enabled;
  fiberDegree_ = degree;
  scoreBlocksByFiber_ = scoreBlocksByFiber;
}

bool CobordismDAG::hasOutputFiber(int node, int outputIndex) const {
  if (node < 0 || node >= static_cast<int>(outputFibers_.size())) return false;
  const auto &f = outputFibers_[static_cast<std::size_t>(node)];
  return outputIndex >= 0 && outputIndex < static_cast<int>(f.size()) &&
         f[static_cast<std::size_t>(outputIndex)].has_value();
}

const BoundaryFiber &CobordismDAG::outputFiber(int node, int outputIndex) const {
  if (!hasOutputFiber(node, outputIndex)) {
    std::string why;
    if (node >= 0 && node < static_cast<int>(fiberRefusals_.size()) &&
        !fiberRefusals_[static_cast<std::size_t>(node)].empty())
      why = " (" + fiberRefusals_[static_cast<std::size_t>(node)] + ")";
    throw std::out_of_range("CobordismDAG::outputFiber: no fiber recorded for node " +
                            std::to_string(node) + " output " + std::to_string(outputIndex) + why);
  }
  return *outputFibers_[static_cast<std::size_t>(node)][static_cast<std::size_t>(outputIndex)];
}

int CobordismDAG::pipedInputCount(int node) const {
  if (node < 0 || node >= static_cast<int>(pipedInputs_.size())) return 0;
  return pipedInputs_[static_cast<std::size_t>(node)];
}

std::string CobordismDAG::fiberRefusal(int node) const {
  if (node < 0 || node >= static_cast<int>(fiberRefusals_.size())) return "";
  return fiberRefusals_[static_cast<std::size_t>(node)];
}

}  // namespace tessera::cobordism
