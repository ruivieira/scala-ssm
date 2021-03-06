name := "scala-ssm"

version := "0.1.1"

organization := "org.ruivieira"

scalaVersion := "2.11.8"

libraryDependencies += "org.scalanlp" %% "breeze" % "0.13"

libraryDependencies += "org.scalanlp" %% "breeze-viz" % "0.13.2"

libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test"

enablePlugins(GhpagesPlugin)

enablePlugins(SiteScaladocPlugin)

git.remoteRepo := "git@github.com:ruivieira/scala-ssm.git"

licenses += ("Apache-2.0", url("https://opensource.org/licenses/Apache-2.0"))

mainClass in assembly := Some("org.ruivieira.ssm.examples.Trend")