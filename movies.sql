-- MySQL Script generated by MySQL Workbench
-- Fri Sep 29 21:48:36 2023
-- Model: New Model    Version: 1.0
-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema movies
-- -----------------------------------------------------
DROP SCHEMA IF EXISTS `movies` ;

-- -----------------------------------------------------
-- Schema movies
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `movies` DEFAULT CHARACTER SET utf8 ;
USE `movies` ;

-- -----------------------------------------------------
-- Table `movies`.`title_basics`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `movies`.`title_basics` ;

CREATE TABLE IF NOT EXISTS `movies`.`title_basics` (
  `tconst` VARCHAR(10) NOT NULL,
  `primary_title` VARCHAR(242) NULL,
  `start_year` YEAR(4) NULL,
  `runtime` INT NULL,
  PRIMARY KEY (`tconst`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `movies`.`ratings`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `movies`.`ratings` ;

CREATE TABLE IF NOT EXISTS `movies`.`ratings` (
  `tconst` VARCHAR(10) NOT NULL,
  `average_rating` FLOAT NULL,
  `number_of_votes` INT NULL,
  `title_basics_tconst` VARCHAR(10) NULL,
  PRIMARY KEY (`tconst`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `movies`.`genres`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `movies`.`genres` ;

CREATE TABLE IF NOT EXISTS `movies`.`genres` (
  `genre_id` INT NOT NULL AUTO_INCREMENT,
  `genre_name` VARCHAR(45) NULL,
  PRIMARY KEY (`genre_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `movies`.`title_basics_has_genres`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `movies`.`title_basics_has_genres` ;

CREATE TABLE IF NOT EXISTS `movies`.`title_basics_has_genres` (
  `tconst` VARCHAR(10) NOT NULL,
  `genre_id` INT NOT NULL,
  PRIMARY KEY (`tconst`, `genre_id`))
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
