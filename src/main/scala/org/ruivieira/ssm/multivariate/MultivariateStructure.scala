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

import breeze.linalg.DenseMatrix
import org.ruivieira.ssm.MatrixUtils
import org.ruivieira.ssm.common.Structure

case class MultivariateStructure(F: DenseMatrix[Double],
                                 G: DenseMatrix[Double],
                                 W: DenseMatrix[Double])
    extends Structure {

  /**
    * Combine two [[MultivariateStructure]] into a single one
    *
    * @param that Another [[MultivariateStructure]]
    * @return A combined [[MultivariateStructure]]
    */
  def +(that: MultivariateStructure): MultivariateStructure = {
    new MultivariateStructure(
      F = MatrixUtils.horizontalCat[Double](F, that.F),
      G = MatrixUtils.blockDiagonal[Double](Array(G, that.G)),
      W = MatrixUtils.blockDiagonal[Double](Array(W, that.W))
    )
  }

}

object MultivariateStructure {

  /**
    * Create a locally constance (underlying mean only) [[MultivariateStructure]]
    *
    * @param d State and observation dimension
    * @param W The state's variance, `W > 0`
    * @return A [[MultivariateStructure]]
    */
  def createLocallyConstant(d: Int,
                            W: DenseMatrix[Double]): MultivariateStructure = {

    val I = DenseMatrix.eye[Double](d)

    new MultivariateStructure(F = I, G = I, W = W)
  }

  def createLocallyLinear(d: Int,
                          W: DenseMatrix[Double]): MultivariateStructure = {

    val I = DenseMatrix.eye[Double](d)
    val zeros = DenseMatrix.zeros[Double](d, d)

    new MultivariateStructure(
      F = DenseMatrix.vertcat(I, zeros),
      G = DenseMatrix.horzcat(DenseMatrix.vertcat(I, zeros),
                              DenseMatrix.vertcat(I, I)),
      W = W)

  }

}
