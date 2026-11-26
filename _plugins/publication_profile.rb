# Derive the lightweight profile assets from the same committed snapshot as the browser.
require "json"

module Jekyll
  class PublicationProfile < Generator
    BIBLIOGRAPHY_FIELDS = %w[title author journal conference volume number pages citation].freeze
    # Samples of the map's Magma palette from the vendored D3.
    MAGMA = %w[000004 0b0924 20114b 3b0f70 57157e 721f81 8c2981 a8327d
               c43c75 de4968 f1605d fa7f5e fe9f6d febf84 fddea0 fcfdbf].freeze

    def generate(site)
      records = JSON.parse(File.read(site.in_source_dir("assets/json/pubs.json"))).fetch("records")
      years = records.map { |record| record.fetch("pub_year").to_i }
      cutoff = [site.time.year, years.max].min - 1
      site.data["recent_publications"] = records.filter_map do |record|
        year = record.fetch("pub_year").to_i
        next if year < cutoff

        {
          "author_pub_id" => record.fetch("author_pub_id"),
          "pub_year" => year,
          "bib_dict" => record.fetch("bib_dict").slice(*BIBLIOGRAPHY_FIELDS)
        }
      end
      site.data["publication_preview"] = preview_points(records, years)
    end

    private

    def fraction(value, bounds)
      low, high = bounds
      high == low ? 0.5 : (value - low).fdiv(high - low)
    end

    def preview_points(records, years)
      x_range = records.map { |record| record.fetch("x") }.minmax
      y_range = records.map { |record| record.fetch("y") }.minmax
      year_range = years.minmax
      max_citations = [records.map { |record| record.fetch("num_citations") }.max, 1].max
      records.zip(years).map do |record, year|
        color_index = (fraction(year, year_range) * (MAGMA.length - 1)).round
        {
          "x" => (40 + 920 * fraction(record.fetch("x"), x_range)).round,
          "y" => (600 - 560 * fraction(record.fetch("y"), y_range)).round,
          "radius" => (10 + 13 * Math.sqrt(record.fetch("num_citations").fdiv(max_citations))).round,
          "color" => MAGMA[color_index]
        }
      end
    end
  end
end
