/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

bool SophonConvolutionQ8Node::verify() const {
  // TBD
   return true;
}

bool SophonFullyConnectedQ8Node::verify() const {
  // TBD
   return true;
}

bool SophonReluQ8Node::verify() const {
  // TBD
   return true;
}

bool SophonMaxPoolQ8Node::verify() const {
  // TBD
   return true;
}

bool SophonAvgPoolQ8Node::verify() const {
  // TBD
   return true;
}

void SophonLoadInst::verify() const {
  // TBD
}

void SophonStoreInst::verify() const {
  // TBD
}

#if 0
bool SophonFakeParallelStartInst::verify() const {}
bool SophonFakeParallelEndInst::verify() const {}

bool TL_AllocInst::verify() const {}
bool SophonDeallocLocalTensorInst::verify() const {}

bool SophonLoadStrideInst::verify() const {}
bool SophonLoadInst::verify() const {}
bool SophonStoreStrideInst::verify() const {}
bool SophonStoreInst::verify() const {}

bool SophonLocalMacInst::verify() const {}
bool SophonLocalMaxInst::verify() const {}
bool SophonLocalCmpInst::verify() const {}
bool SophonLocalMulInst::verify() const {}

bool SophonLocalReshapeInst::verify() const {}
#endif
