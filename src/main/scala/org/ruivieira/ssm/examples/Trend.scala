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

package org.ruivieira.ssm.examples

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.plot.{Figure, plot}
import org.ruivieira.ssm.multivariate.{
  MultivariateGenerator,
  MultivariateStructure
}
import org.ruivieira.ssm.univariate.StateGenerator

/**
  * Created by rui on 12/07/2017.
  */
object Trend {

  def main(args: Array[String]): Unit = {

    val dimension = 20

    val I = DenseMatrix.eye[Double](dimension)
    val zeros = DenseMatrix.zeros[Double](dimension, dimension)

    val W = DenseMatrix.horzcat(DenseMatrix.vertcat(I, zeros),
                                DenseMatrix.vertcat(zeros, I))

    val structure = MultivariateStructure.createLocallyLinear(dimension, W)

    val states = StateGenerator.states(
      nobs = 5000,
      state0 = DenseVector.fill[Double](2 * dimension, 0.0),
      structure = structure)

    val observations =
      MultivariateGenerator.gaussian(states = states,
                                     structure = structure,
                                     V = DenseMatrix.eye[Double](dimension))
    observations.foreach(y => println(y))
  }

}
