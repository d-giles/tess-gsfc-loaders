#!/usr/bin/env python

"""Tests for `tess_gsfc_loaders` package."""


import unittest
from click.testing import CliRunner

from tess_gsfc_loaders import tess_gsfc_loaders
from tess_gsfc_loaders import cli


class TestTess_gsfc_loaders(unittest.TestCase):
    """Tests for `tess_gsfc_loaders` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'tess_gsfc_loaders.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
