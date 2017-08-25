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

import breeze.linalg.DenseVector
import breeze.numerics.exp
import breeze.stats.distributions.{Binomial, Gaussian, Multinomial, Poisson}
import org.ruivieira.ssm.{MathUtils, State}
import org.ruivieira.ssm.common.Structure

object UnivariateGenerator {

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
               V: Double): Vector[Double] = {

    states.map { state =>
      val lambda = structure.F.t * state
      Gaussian(mu = lambda(0), sigma = V).sample()
    }
  }

  /**
    * Generates an [[scala.Array]] representing discrete observations
    * provided by the generated states and structure
    *
    * @param states    The state sequence, an [[Array]] of [[DenseVector]]
    * @param structure The a [[org.ruivieira.ssm.univariate.UnivariateStructure]] for this model
    * @return [[scala.Array]] of observations
    */
  def poisson(states: Vector[State[Double]],
              structure: UnivariateStructure): Vector[Int] = {

    states.map { state =>
      val lambda = structure.F.t * state
      Poisson(mean = exp(lambda(0))).sample()
    }

  }

  /**
    * Generates an [[scala.Array]] representing discrete (categorical) observations
    * provided by the generated states and structure
    *
    * @param categories The number of categories (`categories = 1` represents binary observations)
    * @param states     The state sequence, an [[Array]] of [[DenseVector]]
    * @param structure  The a [[org.ruivieira.ssm.univariate.UnivariateStructure]] for this model
    * @return [[scala.Array]] of observations
    */
  def binomial(categories: Int,
               states: Vector[State[Double]],
               structure: UnivariateStructure): Vector[Int] = {

    states.map { state =>
      val lambda = structure.F.t * state
      Binomial(n = categories, p = MathUtils.ilogit(lambda(0))).sample()
    }

  }

  def multinomial(n: Int,
                  states: Vector[State[Double]],
                  structure: UnivariateStructure) = {

    states.map { state =>
      val lambda = (structure.F.t * state).map(MathUtils.ilogit)
      Multinomial(lambda).sample(n)
    }
  }

}
