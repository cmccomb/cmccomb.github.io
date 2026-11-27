# Keep browser caches in sync when frontend code changes independently of a site release.
require "digest"

module Jekyll
  class FrontendRevision < Generator
    def generate(site)
      paths = %w[css js].flat_map { |type| Dir[site.in_source_dir("assets/#{type}/*.#{type}")] }.sort
      hashes = paths.map { |path| Digest::SHA256.file(path).hexdigest }
      site.data["frontend_revision"] = Digest::SHA256.hexdigest(hashes.join)[0, 12]
    end
  end
end
