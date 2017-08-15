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
import org.ruivieira.ssm.MatrixUtils
import org.ruivieira.ssm.common.Structure

case class UnivariateStructure(F: DenseMatrix[Double],
                               G: DenseMatrix[Double],
                               W: DenseMatrix[Double])
    extends Structure {

  /**
    * Combine two [[UnivariateStructure]] into a single one
    *
    * @param that Another [[UnivariateStructure]]
    * @return A combined [[UnivariateStructure]]
    */
  def +(that: UnivariateStructure): UnivariateStructure = {
    new UnivariateStructure(
      F = MatrixUtils.horizontalCat[Double](F, that.F),
      G = MatrixUtils.blockDiagonal[Double](Array(G, that.G)),
      W = MatrixUtils.blockDiagonal[Double](Array(W, that.W))
    )
  }

}

object UnivariateStructure {

  /**
    * Create a locally constance (underlying mean only) [[UnivariateStructure]]
    *
    * @param W The state's variance, `W > 0`
    * @return A [[UnivariateStructure]]
    */
  def createLocallyConstant(W: Double = 1.0): UnivariateStructure = {

    new UnivariateStructure(F = DenseMatrix.eye[Double](1),
                            G = DenseMatrix.eye[Double](1),
                            W = DenseMatrix.eye[Double](1) * W)
  }

  /**
    * Create a locally linear (underlying mean and trend) [[UnivariateStructure]]
    *
    * @param W The state's variance, `W > 0`
    * @return A [[UnivariateStructure]]
    */
  def createLocallyLinear(W: DenseMatrix[Double] = DenseMatrix.eye[Double](2))
    : UnivariateStructure = {
    val F = DenseMatrix.zeros[Double](2, 1)
    F(0, 0) = 1
    val G = DenseMatrix.ones[Double](2, 2)
    G(1, 0) = 0
    new UnivariateStructure(F = F, G = G, W = W)

  }

  /**
    * Create a purely seasonal [[UnivariateStructure]]
    *
    * @param period The seasonality period
    * @return A [[UnivariateStructure]]
    */
  def createSeasonal(period: Int): UnivariateStructure = {
    val F = DenseMatrix.zeros[Double](period - 1, 1)
    F(0, 0) = 1
    val G = DenseMatrix.zeros[Double](period - 1, period - 1)
    (0 until G.cols).foreach(n => G(0, n) = -1)
    (1 until G.rows).foreach(n => G(n, n - 1) = 1)
    new UnivariateStructure(F = F,
                            G = G,
                            W = DenseMatrix.eye[Double](period - 1))
  }

  /**
    * Builds a Fourier harmonic
    * @param period The seasonality period
    * @param harmonic The harmonic number
    * @return A Fourier harmonic block as a [[DenseMatrix]]
    */
  private def buildHarmonic(period: Int, harmonic: Int): DenseMatrix[Double] = {
    val om = 2.0 * Math.PI / period.toDouble
    val H = DenseMatrix.eye[Double](2) * Math.cos(harmonic.toDouble * om)
    H(0, 1) = Math.sin(harmonic.toDouble * om)
    H(1, 0) = -H(0, 1)
    H
  }

  /**
    * Create seasonal [[UnivariateStructure]] using a Fourier representation
    *
    * @param period The seasonality period
    * @param harmonics The number of Fourier harmonics
    * @return A [[UnivariateStructure]]
    */
  def createCyclical(period: Int, harmonics: Int): UnivariateStructure = {
    val om = 2.0 * Math.PI / period.toDouble

    val H1 = buildHarmonic(period, harmonic = 1)
    val G = if (harmonics > 1) {
      val h = new Array[DenseMatrix[Double]](harmonics)
      h(0) = H1.copy
      (1 until harmonics) foreach { i =>
        h(i) = buildHarmonic(period, harmonic = i + 1)
      }
      MatrixUtils.blockDiagonal(h)
    } else {
      H1
    }
    val dim = G.rows
    val F = DenseMatrix.zeros[Double](dim, 1)
    F(::, 0) := DenseVector[Double](
      Array.fill(harmonics)(Array(1.0, 0.0)).flatten)
    new UnivariateStructure(F = F, G = G, W = DenseMatrix.eye[Double](dim))
  }
}
