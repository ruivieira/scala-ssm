/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ruivieira.ssm

import breeze.linalg.DenseVector
import breeze.stats.distributions.MultivariateGaussian

object StateGenerator {

  /**
    * Generates an [[Array]] of states corresponding to a DGLM with
    * the specified [[Structure]].
    *
    * @param nobs The number of observations to generate
    * @param structure The model's [[Structure]]
    * @param state0 The initial state as a [[DenseVector]]
    * @return An [[Array]] of [[DenseVector]] states
    */
  def states(nobs: Int,
             structure: Structure,
             state0: DenseVector[Double]): Array[DenseVector[Double]] = {

    val states = Array.ofDim[DenseVector[Double]](nobs + 1)
    states(0) = state0

    (1 to nobs).foreach(
      t =>
        states(t) = MultivariateGaussian(mean = structure.G * states(t - 1),
                                         covariance = structure.W).sample())

    // drop the state prior
    states.drop(1)
  }
}
