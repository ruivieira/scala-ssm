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

package org.ruivieira.ssm.test

import breeze.linalg.DenseVector
import org.ruivieira.ssm.univariate.{
  StateGenerator,
  UnivariateGenerator,
  UnivariateStructure
}
import org.scalatest.{FlatSpec, Matchers}

class UnivariateGeneratorTests extends FlatSpec with Matchers {

  "Generated state list" should "have correct size" in {

    val nobs = 1000

    val structure = UnivariateStructure.createLocallyConstant()

    val states = StateGenerator.states(nobs = nobs,
                                       structure = structure,
                                       state0 = DenseVector.zeros[Double](1))

    assert(states.size == nobs)

  }

  "Generated Gaussian observations" should "have the correct number of elements" in {

    val nobs = 1000

    val structure = UnivariateStructure.createLocallyConstant()

    val states = StateGenerator.states(nobs = nobs,
                                       structure = structure,
                                       state0 = DenseVector.zeros[Double](1))

    val observations =
      UnivariateGenerator.gaussian(states = states,
                                   structure = structure,
                                   V = 1.2)

    assert(observations.size == nobs)

  }

  "Generated Poisson observations" should "have the correct number of elements" in {

    val nobs = 100

    val structure = UnivariateStructure.createLocallyConstant()

    val states = StateGenerator.states(nobs = nobs,
                                       structure = structure,
                                       state0 = DenseVector.zeros[Double](1))

    val observations =
      UnivariateGenerator.poisson(states = states, structure = structure)

    assert(observations.size == nobs)

  }

  "Generated Binomial observations" should "have the correct number of elements" in {

    val nobs = 1000

    val structure = UnivariateStructure.createLocallyConstant()

    val states = StateGenerator.states(nobs = nobs,
                                       structure = structure,
                                       state0 = DenseVector.zeros[Double](1))

    val observations =
      UnivariateGenerator.binomial(states = states,
                                   structure = structure,
                                   categories = 1)

    assert(observations.size == nobs)

  }
}
