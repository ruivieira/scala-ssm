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

package org.ruivieira.ssm.multivariate

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.exp
import breeze.stats.distributions._
import org.ruivieira.ssm.{MathUtils, State}
import org.ruivieira.ssm.common.Structure

object MultivariateGenerator {

  /**
    * Generates an [[scala.Array]] representing continuous observations
    * provided by the generated states and structure
    *
    * @param states    The state sequence, an [[Array]] of [[DenseVector]]
    * @param structure The a [[org.ruivieira.ssm.univariate.UnivariateStructure]] for this model
    * @param V         Observation's variance, [[scala.Double]], `V` > 0
    * @return [[scala.Array]] of observations
    */
  def gaussian(states: Vector[State[Double]],
               structure: Structure,
               V: DenseMatrix[Double]): Vector[DenseVector[Double]] = {

    states.map { state =>
      val lambda = structure.F.t * state
      MultivariateGaussian(mean = lambda, covariance = V).sample()
    }
  }

  def multinomial(n: Int,
                  states: Vector[State[Double]],
                  structure: Structure): Vector[DenseVector[Int]] = {

    states.map { state =>
      val lambda = (structure.F.t * state).map(MathUtils.ilogit)
      DenseVector(Multinomial(lambda).sample(n).toArray)
    }
  }

  def poisson(states: Vector[State[Double]],
              structure: Structure): Vector[DenseVector[Int]] = {

    states.map { state =>
      val lambda = exp(structure.F.t * state)
      DenseVector(lambda.data.map(mean => Poisson(mean).sample))
    }
  }

}
