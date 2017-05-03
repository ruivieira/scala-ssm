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

import breeze.linalg.DenseMatrix
import breeze.storage.Zero

import scala.reflect.ClassTag

object MatrixUtils {

  /**
    * Builds a block diagonal matrix
    *
    * @param matrices  An [[Array]] of matrices
    * @return A [[DenseMatrix]] obtained by combining `matrices` as block diagonals
    */
  def blockDiagonal[@specialized(Double, Int, Float, Long) T: ClassTag: Zero](
      matrices: Array[DenseMatrix[T]]): DenseMatrix[T] = {

    val width = matrices.map(_.cols).sum
    val height = matrices.map(_.rows).sum

    val result = DenseMatrix.zeros[T](height, width)

    var rstart = 0
    var cstart = 0

    matrices.foreach { matrix =>
      val r = matrix.rows
      val c = matrix.cols

      (0 until r).foreach { rr =>
        (0 until c).foreach { cc =>
          result(rr + rstart, cc + cstart) = matrix(rr, cc)
        }
      }

      rstart += r
      cstart += c
    }

    result
  }

  /**
    * Provides horizontal concatenation for single-row matrices
    * @param matrices The [[DenseMatrix]] array to concatenate
    * @tparam T The [[DenseMatrix]] type
    * @return A single row [[DenseMatrix]]
    */
  def horizontalCat[@specialized(Double, Int, Float, Long) T: ClassTag](
      matrices: DenseMatrix[T]*): DenseMatrix[T] = {

    val values = matrices.flatMap(_.toArray).toArray

    new DenseMatrix[T](values.length, 1, values)
  }

}
