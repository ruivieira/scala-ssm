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

package org.ruivieira.ssm.univariate

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.MultivariateGaussian
import org.ruivieira.ssm.State
import org.ruivieira.ssm.common.Structure

object StateGenerator {

  /**
    * Generates an [[Array]] of states corresponding to a DGLM with
    * the specified [[UnivariateStructure]].
    *
    * @param nobs The number of observations to generate
    * @param structure The model's [[UnivariateStructure]]
    * @param state0 The initial state as a [[DenseVector]]
    * @return An [[Array]] of [[DenseVector]] states
    */
  def states(nobs: Int,
             structure: Structure,
             state0: State[Double]): Vector[State[Double]] = {

    val states = Array.ofDim[State[Double]](nobs + 1)
    states(0) = state0

    (1 to nobs).foreach { t =>
      states(t) = MultivariateGaussian(mean = structure.G * states(t - 1),
                                       covariance = structure.W).sample()
    }

    // drop the state prior
    // TODO: Make a recursive solution
    states.drop(1).toVector
  }

  /**
    * Generates an [[Array]] of states corresponding to an AR(1) DGLM
    *
    * @param nobs The number of observations to generate
    * @param alpha First auto-regressive parameter
    * @param beta Second auto-regressive parameter
    * @param state0 The initial state as a [[DenseVector]]
    * @param W The state's variance, `W > 0`
    * @return An [[Array]] of [[DenseVector]] states
    */
  def ar1(nobs: Int,
          alpha: Double,
          beta: Double,
          state0: State[Double],
          W: Double): Vector[State[Double]] = {

    val states = Array.ofDim[State[Double]](nobs + 1)
    val variance = DenseMatrix.eye[Double](1) * W

    states(0) = state0
    (1 to nobs).foreach(
      t =>
        states(t) = MultivariateGaussian(mean = states(t - 1) * beta + alpha,
                                         covariance = variance).sample())

    // drop the state prior
    // TODO: Make a recursive solution
    states.drop(1).toVector

  }
}
