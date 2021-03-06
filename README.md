# scala-ssm [![Build Status](https://travis-ci.org/ruivieira/scala-ssm.svg?branch=master)](https://travis-ci.org/ruivieira/scala-ssm)

A Scala library for State-Space Models (SSM).

At the moment, supports:

 - Univariate Dynamic Generalised Models
 - Multivariate Dynamic Generalised Models
   - Multinomial, Poisson and Gaussian

## Installation

To install the library locally you will need `sbt` and then either download
this repo and run

```
sbt publishLocal
```
or add an `sbt` dependency as
```scala
resolvers += Resolver.bintrayRepo("ruivieira", "maven")
libraryDependencies += "org.ruivieira" %% "scala-ssm" % "0.1.1"
```

## Examples

To use the example notebook, you must have installed the library locally (as above) and have the [Jupyter Scala](https://github.com/jupyter-scala/jupyter-scala) kernel installed.

Provided you have those, simply run

```
jupyter notebook
```
