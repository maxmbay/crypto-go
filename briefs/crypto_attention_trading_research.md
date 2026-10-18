# Crypto trading from attention, narratives, and liquidity

Research addendum for Maxwell Bay. Sources checked October 18, 2026.

## Assessment

A crypto-specific variant is viable as a serious modeling and autonomous-execution project. A profitable edge remains to be demonstrated. The strongest research question is:

> When does a change in attention predict additional buying demand and executable returns, and when does it indicate that a move is already exhausted or artificially promoted?

I recommend beginning with liquid, exchange-listed spot assets, including established meme coins, and examining signals over hours to days. A second research environment can address newly launched on-chain tokens, where liquidity, coordinated ownership and failure are central modeling targets. These environments have different populations, costs and execution mechanisms; success in one should not be treated as evidence for the other.

The main contribution would be a time-respecting, cost-aware model of returns and liquidity that makes calibrated decisions about whether to trade. Sentiment, multimodal encoders, Hawkes processes, and on-chain risk classifiers already have substantial prior art.

The first collection plan uses free public feeds: 4chan, Bluesky's live stream, DEX Screener and exchange data, with Google Trends/GDELT for broader context. Paid social-history services are optional extensions after a coverage audit.

## 1. What has already been done

| Work | Demonstrated contribution | What remains to establish for this project |
| --- | --- | --- |
| Liu and Tsyvinski, *Risks and Returns of Cryptocurrency*, RFS 2021 / earlier working paper | Historical momentum and investor-attention measures forecast crypto returns in their sample. [S1] | Persistence in a current, executable universe under our account's costs |
| Maitre, Pugachyov and Weigert, JBF 2025 | Abnormal Twitter attention predicts next-day cross-sectional returns; user ticker posts carry different information from official project posts. [S2] | Whether source-aware features add value with present-day data and execution |
| Ortu et al., ML with Applications 2022 | Topics, sentiment and price events are studied with Hawkes processes. [S3] | A modern forecasting and trading test; temporal excitation alone does not identify causality |
| CoinCLIP / CoinVibe, 2024 preprint and subsequent conference work | Combines frozen CLIP representations of logos and descriptions with community comments. Its viability label is DEX listing. [S4] | Returns available to an actual buyer at a defined decision time |
| MELT, 2026; earlier version called MemeTrans | Behavioral launch traces and coordinated-account features support risk classification. It already uses a chronological test split. [S5] | Profitable selection, execution realism and broader populations |
| MemeChain, 2026 | Cross-chain token, financial, website, logo and social-link metadata support forensic research. [S6] | An extended historical social stream and executable market paths |

The published attention results do not imply that more online discussion is always bullish. A 2026 working paper by Chu and Tsiakas separates news novelty from repeated dissemination and reports different return relationships. Treat that finding as preliminary, but it motivates testing what the content adds beyond its volume. [S7]

### Important distinctions in the closest meme-coin work

**CoinCLIP's target:** Reaching a DEX listing is different from making money after entry. Its reported 80/10/10 split does not by itself establish a prospective early-warning trading evaluation. Comments and likes need an explicit as-of cutoff before they can be used as historical trading inputs. This is a limitation to resolve in our use of the resource, not a claim that the authors intended to validate a trading bot. [S4]

**MELT's population and economics:** Its 41k-plus launches span December 2024–March 2025 and include successful DEX migrations. Post-migration traces cover the first hour. The selection simulation buys at migration prices and samples later sale times; model-guided portfolios still lose money, though less than random selection. A useful defensive classifier therefore does not automatically supply a profitable policy. [S5]

**MemeChain's scope:** Its 34,988-token snapshot spans Ethereum, BSC, Solana and Base, with lifecycle observations extending from October 2024 into January 2025. Website archives are single-time snapshots; social links are not a complete historical post corpus. It is a research starting point rather than a complete trading dataset. [S6]

**Manipulation evidence:** Mongardini and Mei report manipulation indicators in 82.8% of the high-return tokens in their studied cohort. This is a conditional finding about that sample, not a rate for every meme coin. It makes coordinated promotion and artificial volume plausible explanations that our features must distinguish. [S8]

## 2. The most defensible advances

These are proposed contributions relative to the work reviewed. A first-ever novelty claim would require a further targeted literature review after the task and dataset are fixed.

### A. Predict remaining opportunity after attention arrives

Build an attention baseline from past posting activity and past price moves, then examine whether unexpected changes in discussion predict subsequent demand or returns.

Useful inputs include:

- New unique authors and source diversity, rather than post count alone.
- Semantic novelty relative to earlier discussion.
- Repeated or near-duplicate promotional language.
- Whether attention is broadening beyond the project's existing audience.
- The ordering of attention, trading volume, buyer participation and price changes.
- Whether the market has already moved substantially before the signal becomes observable.

A residual from an attention model is a predictive feature, not proof of an exogenous information shock. Shared news, hidden promoters and common market conditions can drive both attention and prices. Assess incremental forecasting value after controlling for price/volume history and market factors.

The decisive comparison is a price-only model against price plus simple attention counts, followed by a model with source, novelty and timing information. This isolates whether language understanding contributes something more than detecting a price spike indirectly.

### B. Forecast returns together with liquidity and failure

For small tokens, a displayed mark is an incomplete target. Define outcomes at a specified position size and exit policy: actual or conservatively simulated purchase cost, sale proceeds, fees and execution failures.

Use a joint model with a shared latent state to estimate:

- The distribution of holding-period returns.
- Changes in spread, available depth or pool liquidity.
- The probability and timing of a sharp liquidity or price failure.
- The size-dependent distribution of proceeds from exiting.

For an early on-chain token, a heavy-tailed return component plus a failure/jump component is more plausible than a narrow Gaussian forecast. Bankruptcy-like outcomes or failed exits require explicit treatment. Separate classifier scores for risk and expected returns can be a first baseline; a joint distribution should earn its complexity by improving decisions.

Survival prediction is already contemplated by existing dataset authors. The proposed advance is dynamic prediction from observable prefixes, coupled to position-size-dependent returns and an executable policy.

### C. Learn propagation between related tokens

Test whether activity in one asset predicts later activity in another after controlling for broad crypto and chain-level moves. Candidate relations include shared topics, overlapping promoters, contemporaneously observable wallet patterns and historical response timing.

Use soft narrative memberships: an asset can participate in several themes, and a new narrative can emerge. Construct memberships using only information available by the decision date. A cluster fitted to the entire eventual episode would import hindsight.

A temporal graph or set transformer can test whether learned interactions outperform simple market, sector and correlation features. The graph is justified only if it adds out-of-sample information. The direction and timing of predictive relationships should be evaluated separately from causal claims.

### D. Test transfer across episodes and entities

Use future-time tests and additional holdouts for newly seen narratives, creators and coordinated wallet groups. Where groups are uncertain, perform sensitivity analysis over alternative linkage rules.

Repeated images, project descriptions or account clusters can make a model look general while it recognizes familiar campaigns. Holding out an episode or entity is a harder and more informative test than allocating closely related tokens to both training and testing sets.

For wallet models, do not equate one address with one person or assume every shared funding source implies common ownership. The model should represent uncertainty in these relationships.

## 3. Two practical variants

| Choice | Exchange-listed spot assets | New on-chain meme coins |
| --- | --- | --- |
| Initial objective | Predict subsequent returns using attention and market state | Predict size-dependent returns, liquidity and failure after token creation/migration |
| Observation scale | Hourly data; slower entry/exit policy | Irregular transaction events or short bins |
| Initial horizon | Fix one horizon, such as 24 hours, before adding others | Fix a decision event and short holding interval consistent with available traces |
| Research strengths | Longer histories, cleaner execution interfaces, broader regimes | Rich behavioral traces and substantial model-design scope |
| Main limitations | Social history and small signals relative to fees | Narrow datasets, fast adversarial behavior, exit realism and protocol changes |
| Recommended role | First autonomous trading prototype | Separate, higher-complexity research extension |

Begin with a historically selected universe on one accessible exchange. Eligibility uses the listing state, trailing liquidity, spread and data completeness known then. Include BTC/ETH and broad market state as explanatory context. Public derivatives information such as funding, open interest and liquidations may add context even when the strategy trades spot only.

For the first economic experiment, define one entry schedule and a fixed holding policy. If a later policy reassesses every four hours but targets a 24-hour opportunity, measure actual holdings and fills in the ledger; overlapping forecast labels are not portfolio P&L.

Continuous data collection does not require continuous trading. The policy can hold an existing position or cash when the marginal expected benefit of changing positions is too small.

## 4. Data access and what is missing

| Requirement | Candidate source | Access and interpretation |
| --- | --- | --- |
| Historical exchange bars | Kraken downloadable OHLCVT; exchange-specific APIs | Kraken publishes bulk history, while its OHLC REST endpoint only returns the latest 720 observations. Bulk files and live collection need a common schema. [S9, S10] |
| Historical depth, trades, funding and liquidations | Tardis | Recorded exchange feeds, with exchange/symbol coverage dates and incident information. Useful for execution research; access is not equivalent to an unlimited free dataset. [S11] |
| Social attention aggregates | Santiment | Historical social metrics across curated sources; inspect metric versions, history and real-time restrictions. Aggregate access is different from access to underlying raw text. [S12] |
| Optional paid post history | X API or a licensed corpus | Consider after a coverage audit identifies a gap that justifies paid access. [S13] |
| Crypto language baseline | CryptoBERT | A domain-adapted sentiment model trained on over 3.2 million posts; sentiment labels are not realized-return labels. [S14] |
| Early launch behavior | MELT author repository | Parsed transaction/bundle downloads and a precomputed feature file are linked. Full event modeling requires the transaction traces rather than only the feature table. [S15] |
| Token websites and branding | MemeChain on Zenodo | Registry, metadata, HTML and logo resources; inspect collection dates and coverage. [S16] |
| DEX pool market histories | CoinGecko on-chain API | Pool-specific OHLCV, with historical access dependent on plan and pool tracking start. [S17] |
| Discovery and live snapshots | DEX Screener API | Current profiles, pairs and related endpoints are useful for a forward collector. Do not infer a complete historical panel from a current discovery API. [S18] |

I did not find one ready-to-train public resource containing all required social, market, liquidity and historical-universe inputs. Data acquisition is likely to be an early bottleneck.

### Free-source first collection plan

The initial prototype prioritizes **4chan + Bluesky live data + DEX Screener + exchange prices**. Google Trends exports and GDELT add broader attention context. A paid historical X corpus is optional.

| Source | What to collect | Access qualification |
| --- | --- | --- |
| 4chan, initially /biz/ | Thread catalogs, token mentions, reply activity, repeated phrases and images | Official read-only JSON API. Respect its one-request-per-second limit and minimum ten-second thread-update interval. Start a forward archive; do not assume complete historical coverage or stable author identities. [S21] |
| Bluesky Jetstream | Public posts and relevant engagement events | The public live stream requires no authentication and is unmetered. Hosted archive replay has separate authentication and metering. Audit actual coverage of the intended crypto communities. [S22] |
| DEX Screener | Profiles, ads, boosts, community takeovers and pair metrics | Free API with endpoint-specific limits. Purchased boosts measure promotion; keep them distinct from independent discussion growth. Curated metas can seed theme discovery with their observation dates recorded. [S18, S23] |
| Google Trends | Search-interest series | Free CSV exports support initial historical work. The official programmatic API remains limited-access alpha. [S24] |
| GDELT | News metadata and theme coverage | Free open metadata and archives for broader awareness signals. [S25] |

For each eligible token and hourly interval, derive mention counts, discussion acceleration, distinct observable authors or thread participation, semantic novelty, near-duplicate promotion, cross-source arrival times and explicit promotion/takeover flags. Track source coverage and outages. Use chain/contract identifiers where possible and retain uncertainty for ambiguous tickers.

Begin with a coverage and latency pilot: measure eligible-asset coverage, signal delay, duplication and storage requirements while building a reliable forward dataset. Trading value requires a longer independent evaluation. Historical market data permits immediate price-only baselines while the community dataset accumulates.

Farcaster is an optional open-data source, but full replication needs infrastructure: its current node guide specifies 16 GB RAM and 2 TB storage. Validate a small access route before committing to a full node. [S26]

Free API access and model-development rights are separate questions. Telegram's API terms expressly restrict ML development and deployment; Reddit also imposes training-permission requirements. Keep these outside the default automated training pipeline unless the intended use is appropriately authorized. [S27, S28]

Storage, processing and optional hosted services still have resource costs. No single reviewed free resource supplies the complete historically aligned panel. Paid history can be added if a measured coverage gap justifies it. As checked October 18, 2026, X lists Post reads at $0.005 per returned resource, equivalent to $5,000 per million before other charges; this is an optional acquisition cost, not an entry requirement. [S13]

### Availability and versioning

Store at least event time, observed/received time, source, source version, asset mapping and any revision identifier. Examples of problematic inputs include a comment's eventual like count, today's follower count, a website captured after a token failed, or a wallet label inferred using later transactions.

Santiment documents that changes in source coverage can revise historical values or introduce artificial discontinuities. Freeze metric versions and audit the source population. Backfilling is not automatically leakage: a newly computed statistic can be valid if it uses only historically observable inputs. Its computation timestamp alone does not settle that question. [S12]

Keep missing intervals distinct from verified zero activity. Bars do not establish an executable exit size. Use historical quotes/pool states or conservative, explicit execution assumptions and prospective measurements.

## 5. A model we can train ourselves in JAX

The first trainable core can be compact:

1. **Market encoder:** A shared recurrent encoder or compact transformer processes returns, volume, volatility, spread/depth and market/chain context.
2. **Attention encoder:** Process rolling counts, unique-source features and cached text embeddings. Preserve event timing and avoid using engagement observed after the decision cutoff.
3. **Cross-asset interaction:** Set attention combines eligible assets and soft narrative representations. Start with simple market-factor conditioning before adding this block.
4. **Distributional outputs:** Heavy-tailed return forecasts and liquidity/risk outputs share a state representation. For the on-chain extension, model event hazards explicitly.
5. **Uncertainty and allocation:** Bayesian output/risk components or time-block ensembles feed a constrained cost-aware allocation rule.

Equinox/Optax can train the forecasting components; NumPyro can support hierarchical effects and modest latent-state models. A conditional flow over a small latent factor vector is a later option if distributional diagnostics warrant it. Reuse the compatible JAX stack described in the original trading brief.

A self-exciting point process is a useful baseline or extension for irregular bursts in posting and trading. Since Hawkes approaches already exist here, the contribution must come from better information, calibration, transfer or economic evaluation. Adding a neural intensity model by itself is a limited advance.

Train returns with likelihood or quantile/proper-score objectives, and train clearly defined event/hazard targets separately. Random-initialization ensemble spread is not automatically a calibrated posterior. Evaluate whether uncertainty changes position sizes in a beneficial way under identical constraints.

Jev or a local classifier can map text into narrow event features: a new announcement, repetitive promotion, a purported listing, or a claim needing corroboration. Our numeric forecasting model determines whether those features predict outcomes. A text model's confident answer is not evidence of a profitable trade.

## 6. Evaluation that would advance prior work

### Fix the experiment before scaling the architecture

Compare:

1. Cash and market/eligible-universe benchmarks.
2. Price/volume momentum and reversion baselines.
3. LightGBM with market state and simple attention counts.
4. A temporal model using the same inputs.
5. Source/novelty features and cross-asset narrative interactions.
6. The best forecaster with and without uncertainty-aware allocation.

Keep universe, costs, liquidity assumptions, risk limits and model-selection rules identical for each relevant comparison. Include strong tabular baselines for wallet/launch tasks.

Use chronological walk-forward tests with matured labels and non-overlapping development/final evaluation periods. Block-bootstrap dates or episodes while retaining their cross-asset dependence. A transaction dataset with hundreds of millions of rows may still contain relatively few independent market episodes.

Add entity/narrative holdouts, later market regimes, adverse cost assumptions, signal delays and failed or partial exits. A future launch cohort should be eligible without requiring knowledge of its later success. If the prediction time is migration, conditioning on observed migration is appropriate; if it is token creation, selecting only future migrants is not.

### Economic outcomes

Use fills and cash flows to compute returns after costs, turnover, drawdowns, tail losses and capacity. Measure incremental performance against market and momentum exposures, not just outright profit during a crypto rally. Show calibration on the traded subset as well as the full candidate population.

For a DEX strategy, costs include pool/route fees, size-dependent impact, network/priority fees, inclusion delay, failed transactions and adverse execution. For a CEX strategy, include actual account-tier maker/taker fees, spread, fill probability and adverse selection. A passive limit order is not a free guaranteed fill. Coinbase explicitly states that applicable rates depend on product, tier and liquidity-taking/providing status. [S19]

An illustrative 0.5% round-trip cost already exceeds a 0.2% forecast gain. These are hypothetical values, not a quote for an exchange. A recent Bitcoin study likewise illustrates that an apparently positive gross forecast can become an unprofitable high-turnover policy; its selected results remain a preprint rather than an independently established live edge. [S20]

Abstention needs an economic baseline: avoiding a loss can be valuable, but cash is also an available action. Better classification or smaller losses should not be relabeled as positive investment returns.

### What would make the project distinctive

A credible contribution would demonstrate one of the following:

- Source diversity and semantic novelty predict returns beyond simple attention counts and market history.
- Learned narrative relationships transfer to unseen episodes and assets.
- Joint liquidity/failure modeling produces better size-dependent decisions than return-only forecasts.
- Calibrated uncertainty improves realized net utility or tail behavior under otherwise identical rules.
- An open evaluation dataset/protocol exposes where published classification success fails to translate into tradable outcomes.

Report negative results if the advanced components fail to improve the appropriate baselines. That is a substantive result in this problem space.

## 7. A concrete first build

**First data audit:** Select one accessible spot venue and a small historical universe. Begin forward collection from 4chan, Bluesky and DEX Screener, and assess Google Trends/GDELT as context. Check listings/delistings, source coverage, reception latency, fees and historical spread/depth proxies. Require a reproducible as-of join before training the advanced model.

**First modeling experiment:** Fix a 24-hour holding policy. Compare LightGBM on market data, then simple abnormal attention, then source/novelty features. Add a compact shared temporal model and cross-asset attention only after the earlier comparisons work. Use a preserved later interval for the final comparison.

**First execution experiment:** Run the frozen candidate in shadow/paper mode. Record actual order-book conditions, candidate orders and simulated/fill outcomes. Independent limits govern capital, position and narrative/chain concentration, turnover, liquidity participation, stale inputs and unknown order state. A continuously running collector should recover and reconcile after interruptions.

**On-chain research extension:** Reproduce the published launch-risk baseline using the author data. Then test size-dependent returns and dynamic updating from observed prefixes. Extend the population and time coverage before claiming transfer to today's new launches. This is a separately scoped experiment rather than an assumption that CEX and launchpad strategies share one predictive distribution.

The first two weeks should produce a data feasibility assessment and simple cost-aware baseline, not a claim of alpha. The deliverable worth pursuing is a system that can explain which information improved a forecast, whether it justified a trade, and how that conclusion survived a realistic evaluation.

## Sources

Primary papers, author repositories, dataset records, and official documentation. Prices and access descriptions reflect the research date; this review did not download and audit every underlying dataset byte.

- **S1.** [Liu and Tsyvinski, Risks and Returns of Cryptocurrency](https://www.nber.org/papers/w24877); [author's publication list identifying RFS 2021](https://www.yukunliu.com/research/).
- **S2.** [Maitre, Pugachyov and Weigert, Social media-based attention and the cross-section of cryptocurrency returns, JBF 2025](https://www.sciencedirect.com/science/article/abs/pii/S0378426625001384); [author working-paper version](https://cfr-cologne.de/download/workingpaper/cfr-25-02.pdf).
- **S3.** [Ortu et al., Cryptocurrency ecosystems and social media environments: An empirical analysis through Hawkes' models and natural language processing](https://bura.brunel.ac.uk/handle/2438/23708).
- **S4.** [CoinCLIP paper](https://arxiv.org/abs/2412.07591); [methods and label definitions](https://arxiv.org/html/2412.07591v2); [code/data link supplied by authors](https://github.com/hwlongCUHK/CoinCLIP).
- **S5.** [MELT current paper](https://arxiv.org/abs/2602.13480); [full methods and trading evaluation](https://arxiv.org/html/2602.13480v2).
- **S6.** [MemeChain paper](https://arxiv.org/abs/2601.22185); [dataset scope and limitations](https://arxiv.org/html/2601.22185v1).
- **S7.** [Chu and Tsiakas, Media Coverage and the Cross-Section of Cryptocurrency Returns, 2026 working paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6805101).
- **S8.** [A Midsummer Meme's Dream: Investigating Market Manipulations in the Meme Coin Ecosystem](https://arxiv.org/abs/2507.01963).
- **S9.** [Kraken downloadable historical OHLCVT](https://support.kraken.com/au/articles/360047124832-downloadable-historical-ohlcvt-open-high-low-close-volume-trades-data).
- **S10.** [Kraken OHLC REST API limitations](https://docs.kraken.com/api-reference/market-data/get-ohlc-data).
- **S11.** [Tardis historical feed coverage and incidents](https://docs.tardis.dev/historical-data-details); [downloadable historical datasets](https://docs.tardis.dev/downloadable-csv-files/overview).
- **S12.** [Santiment social sources and history](https://academy.santiment.net/metrics/details/social-data/); [metric versions and historical revisions](https://academy.santiment.net/sanapi/metric-versions/); [historical/realtime restrictions](https://academy.santiment.net/sanapi/historical-and-realtime-data-restrictions/).
- **S13.** [X full-archive search](https://docs.x.com/x-api/posts/search/introduction); [current resource-based pricing](https://docs.x.com/x-api/getting-started/pricing).
- **S14.** [CryptoBERT model card and training corpus description](https://huggingface.co/ElKulako/cryptobert).
- **S15.** [MELT author repository and download instructions](https://github.com/git-disl/MELT); [README](https://github.com/git-disl/MELT/blob/master/README.md).
- **S16.** [MemeChain author dataset record on Zenodo](https://zenodo.org/records/18246856).
- **S17.** [CoinGecko pool OHLCV and historical access](https://docs.coingecko.com/reference/pool-ohlcv-contract-address).
- **S18.** [DEX Screener API reference](https://docs.dexscreener.com/api/reference).
- **S19.** [Coinbase Advanced fees](https://help.coinbase.com/en/coinbase/trading-and-funding/advanced-trade/advanced-trade-fees); [Kraken product and tier fee schedules](https://www.kraken.com/features/fee-schedule).
- **S20.** [Machine Learning-Based Bitcoin Trading Under Transaction Costs: Evidence From Walk-Forward Forecasting, 2026 preprint](https://arxiv.org/abs/2606.00060).
- **S21.** [4chan official read-only API and request rules](https://github.com/4chan/4chan-API/blob/master/README.md); [catalog schema](https://github.com/4chan/4chan-API/blob/master/pages/Catalog.md).
- **S22.** [Bluesky Jetstream public live access](https://bsky.network/docs/jetstream/); [historical replay authentication and metering](https://bsky.network/docs/jetstream-replay/).
- **S23.** [DEX Screener free API](https://docs.dexscreener.com/); [purchased boosts](https://docs.dexscreener.com/boosting); [curated narrative metas](https://docs.dexscreener.com/metas).
- **S24.** [Google Trends CSV exports](https://support.google.com/trends/answer/4365538?hl=en); [official API alpha access](https://developers.google.com/search/apis/trends).
- **S25.** [GDELT open data](https://www.gdeltproject.org/data.html).
- **S26.** [Farcaster data analysis options](https://docs.farcaster.xyz/developers); [Snapchain node requirements](https://snapchain.farcaster.xyz/getting-started).
- **S27.** [Telegram API terms on AI/ML data use](https://core.telegram.org/api/terms).
- **S28.** [Reddit Data API Terms](https://redditinc.com/policies/data-api-terms).
