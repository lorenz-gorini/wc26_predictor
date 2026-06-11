from __future__ import annotations

from wc26_predictor.data.download_public import _parse_group_page


def test_parse_group_page_reads_fixture_boxes() -> None:
    html = """
    <html><body>
      <div class="footballbox">
        <div class="fdate">June 11, 2026 <span>2026-06-11</span></div>
        <div class="ftime">1:00 p.m. <a>UTC−6</a></div>
        <table><tr>
          <th class="fhome"><span itemprop="name">Mexico</span></th>
          <th class="fscore">Match 1</th>
          <th class="faway"><span itemprop="name">South Africa</span></th>
        </tr></table>
        <div class="fright">
          <span itemprop="name address">Estadio Azteca, Mexico City</span>
        </div>
      </div>
      <div class="footballbox">
        <div class="fdate">June 11, 2026 <span>2026-06-11</span></div>
        <div class="ftime">8:00 p.m. <a>UTC−6</a></div>
        <table><tr>
          <th class="fhome"><span itemprop="name">South Korea</span></th>
          <th class="fscore">Match 2</th>
          <th class="faway"><span itemprop="name">Czech Republic</span></th>
        </tr></table>
        <div class="fright">
          <span itemprop="name address">Estadio Akron, Zapopan</span>
        </div>
      </div>
      <div class="footballbox">
        <div class="fdate">June 18, 2026 <span>2026-06-18</span></div>
        <div class="ftime">12:00 p.m. <a>UTC−4</a></div>
        <table><tr>
          <th class="fhome"><span itemprop="name">Czech Republic</span></th>
          <th class="fscore">Match 25</th>
          <th class="faway"><span itemprop="name">South Africa</span></th>
        </tr></table>
        <div class="fright">
          <span itemprop="name address">Mercedes-Benz Stadium, Atlanta</span>
        </div>
      </div>
      <div class="footballbox">
        <div class="fdate">June 18, 2026 <span>2026-06-18</span></div>
        <div class="ftime">7:00 p.m. <a>UTC−6</a></div>
        <table><tr>
          <th class="fhome"><span itemprop="name">Mexico</span></th>
          <th class="fscore">Match 26</th>
          <th class="faway"><span itemprop="name">South Korea</span></th>
        </tr></table>
        <div class="fright">
          <span itemprop="name address">Estadio Akron, Zapopan</span>
        </div>
      </div>
      <div class="footballbox">
        <div class="fdate">June 24, 2026 <span>2026-06-24</span></div>
        <div class="ftime">2:00 p.m. <a>UTC−6</a></div>
        <table><tr>
          <th class="fhome"><span itemprop="name">Czech Republic</span></th>
          <th class="fscore">Match 49</th>
          <th class="faway"><span itemprop="name">Mexico</span></th>
        </tr></table>
        <div class="fright">
          <span itemprop="name address">Estadio Azteca, Mexico City</span>
        </div>
      </div>
      <div class="footballbox">
        <div class="fdate">June 24, 2026 <span>2026-06-24</span></div>
        <div class="ftime">3:00 p.m. <a>UTC−6</a></div>
        <table><tr>
          <th class="fhome"><span itemprop="name">South Africa</span></th>
          <th class="fscore">Match 50</th>
          <th class="faway"><span itemprop="name">South Korea</span></th>
        </tr></table>
        <div class="fright">
          <span itemprop="name address">Estadio BBVA, Monterrey</span>
        </div>
      </div>
    </body></html>
    """

    fixtures = _parse_group_page(html, source_url="https://example.com", group="A")

    assert len(fixtures) == 6
    assert fixtures.loc[0, "date"] == "2026-06-11"
    assert fixtures.loc[0, "utc_offset"] == "UTC-6"
    assert fixtures.loc[0, "match_number"] == 1
    assert fixtures.loc[0, "stadium"] == "Estadio Azteca"
    assert fixtures.loc[0, "city"] == "Mexico City"

