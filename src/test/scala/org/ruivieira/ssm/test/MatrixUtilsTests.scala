package org.ruivieira.ssm.test

import breeze.linalg.DenseMatrix
import org.ruivieira.ssm.MatrixUtils
import org.scalatest.{FlatSpec, Matchers}

/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

class MatrixUtilsTests extends FlatSpec with Matchers {

  "Block diagonal" should "have correct dimensions" in {

    val A = DenseMatrix.eye[Double](5)
    val B = DenseMatrix.eye[Double](2)

    // C = A + B
    val C = MatrixUtils.blockDiagonal[Double](Array(A, B))

    assert(C.cols == 7)
    assert(C.rows == 7)

  }

}
