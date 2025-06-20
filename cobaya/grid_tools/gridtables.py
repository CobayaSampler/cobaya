"""
.. module:: cobaya.grid_tools.gridtables

:Synopsis: make summary pdf with tables for result in a grid (Cobaya version)
:Author: Antony Lewis (based on CosmoMC version of the same code)

"""

import copy
import os

from getdist import paramnames, types
from getdist.mcsamples import loadMCSamples

from cobaya.grid_tools.batchjob_args import BatchArgs
from cobaya.tools import working_directory


def grid_tables(args=None):
    opts = BatchArgs(
        "Make pdf tables from latex generated from getdist outputs",
        importance=True,
        converge=True,
    )
    opts.parser.add_argument("latex_filename", help="name of latex/PDF file to produce")
    opts.parser.add_argument(
        "--limit", type=int, default=2, help="sigmas of quoted confidence intervals"
    )
    opts.parser.add_argument("--all_limits", action="store_true")
    opts.parser.add_argument("--bestfitonly", action="store_true")
    opts.parser.add_argument("--nobestfit", action="store_true")
    opts.parser.add_argument("--no_delta_chisq", action="store_true")
    opts.parser.add_argument(
        "--delta_chisq_paramtag",
        default=None,
        help="parameter tag to give best-fit chi-squared differences",
    )
    opts.parser.add_argument(
        "--changes_from_datatag",
        default=None,
        help="give fractional sigma shifts compared to a given data combination tag",
    )
    opts.parser.add_argument(
        "--changes_from_paramtag",
        default=None,
        help="give fractional sigma shifts compared to a given parameter combination tag",
    )
    opts.parser.add_argument(
        "--changes_adding_data",
        nargs="+",
        default=None,
        help="give fractional sigma shifts when adding given data",
    )
    opts.parser.add_argument(
        "--changes_replacing",
        nargs="+",
        default=None,
        help="give sigma shifts for results with data x, y, z "
        "replacing data y, z.. with x",
    )
    opts.parser.add_argument(
        "--changes_only",
        action="store_true",
        help="Only include results in the changes_replacing set",
    )
    opts.parser.add_argument(
        "--changes_data_ignore",
        nargs="+",
        help="ignore these data tags when mapping to reference for comparison",
    )
    opts.parser.add_argument(
        "--systematic_average",
        action="store_true",
        help="Combine two chains and quote results "
        "for the combination, e.g. as a crude way of "
        "including systematic errors between "
        "likelihood versions. "
        "Used with --changes_replacing or similar.",
    )

    opts.parser.add_argument(
        "--shift_sigma_indep",
        action="store_true",
        help="fractional shifts are relative to the sigma for "
        "independent data (sigma^2=sigma1^2+sigma2^2",
    )
    opts.parser.add_argument(
        "--shift_sigma_subset",
        action="store_true",
        help="fractional shifts are relative to the sigma "
        "for stricly subset data "
        "(sigma^2 = |sigma1^2-sigma2^2|, "
        "regularized to sigma/20)",
    )

    opts.parser.add_argument(
        "--paramNameFile", help=".paramnames file for custom labels for parameters"
    )

    opts.parser.add_argument(
        "--paramList",
        default=None,
        help=".paramnames file listing specific parameters to include (only).",
    )
    opts.parser.add_argument(
        "--blockEndParams",
        default=None,
        help="a semi-colon separated list of parameters marking "
        "the end of distinct parameter blocks "
        "(e.g. physical vs nuisance parmeters, "
        "sampled vs derived)",
    )
    opts.parser.add_argument("--columns", type=int, nargs=1, default=3)
    opts.parser.add_argument("--compare", nargs="+", default=None)

    opts.parser.add_argument("--titles", default=None)  # for compare plots
    opts.parser.add_argument("--forpaper", action="store_true")
    opts.parser.add_argument("--separate_tex", action="store_true")
    opts.parser.add_argument("--header_tex", default=None)
    opts.parser.add_argument("--height", default="13in")
    opts.parser.add_argument("--width", default="12in")

    (batch, args) = opts.parseForBatch(args)

    if args.blockEndParams is not None:
        args.blockEndParams = args.blockEndParams.split(";")

    if args.paramList is not None:
        args.paramList = paramnames.ParamNames(args.paramList)

    def tex_escape_text(string):
        return string.replace("_", "{\\textunderscore}")

    def get_table_lines(content, _referenceDataJobItem=None):
        if _referenceDataJobItem is not None:
            refResults = _referenceDataJobItem.result_marge
        else:
            refResults = None
        return types.ResultTable(
            args.columns,
            [content],
            blockEndParams=args.blockEndParams,
            paramList=args.paramList,
            limit=args.limit,
            refResults=refResults,
            shiftSigma_indep=args.shift_sigma_indep,
            shiftSigma_subset=args.shift_sigma_subset,
        ).lines

    def get_systematic_average_table_lines(jobItem1, jobItem2):
        # if you have two versions of the likelihood with the same data,
        # and don't know which is right,
        # this just crudely adds the samples with equal weight per likelihood
        samps1 = loadMCSamples(
            jobItem1.chainRoot, jobItem=jobItem1, settings=batch.getdist_options
        )
        samps2 = loadMCSamples(
            jobItem2.chainRoot, jobItem=jobItem2, settings=batch.getdist_options
        )
        samps = samps1.getCombinedSamplesWithSamples(samps2)
        return get_table_lines(samps.getMargeStats())

    def param_result_table(jobItem, deltaChisqJobItem=None, _referenceDataJobItem=None):
        if deltaChisqJobItem is not None and deltaChisqJobItem.name == jobItem.name:
            deltaChisqJobItem = None
        if _referenceDataJobItem is not None:
            if (
                args.changes_from_paramtag is None
                and _referenceDataJobItem.normed_data == jobItem.normed_data
                or args.changes_from_paramtag is not None
                and _referenceDataJobItem.name == jobItem.name
            ):
                _referenceDataJobItem = None

        table_lines = []
        caption = []
        jobItem.loadJobItemResults(
            paramNameFile=args.paramNameFile,
            bestfit=not args.nobestfit,
            bestfitonly=args.bestfitonly,
        )
        bf = jobItem.result_bestfit
        if bf is not None:
            caption.append(
                " Best-fit $\\chi^2_{\\rm eff} = " + ("%.2f" % (bf.logLike * 2)) + "$"
            )
            if deltaChisqJobItem is not None:
                bf_ref = deltaChisqJobItem.result_bestfit
                if bf_ref is not None:
                    caption.append(
                        "$\\Delta \\chi^2_{\\rm eff} = "
                        + ("%.2f" % ((bf.logLike - bf_ref.logLike) * 2))
                        + "$"
                    )

        if args.bestfitonly:
            if bf is not None:
                table_lines += get_table_lines(bf)
        else:
            likeMarge = jobItem.result_likemarge
            if likeMarge is not None and likeMarge.meanLogLike is not None:
                caption.append(
                    "$\\bar{\\chi}^2_{\\rm eff} = "
                    + ("%.2f" % (likeMarge.meanLogLike * 2))
                    + "$"
                )
                if deltaChisqJobItem is not None:
                    likeMarge_ref = deltaChisqJobItem.result_likemarge
                    if (
                        likeMarge_ref is not None
                        and likeMarge_ref.meanLogLike is not None
                    ):
                        delta = likeMarge.meanLogLike - likeMarge_ref.meanLogLike
                        caption.append(
                            "$\\Delta\\bar{\\chi}^2_{\\rm eff} = "
                            + ("%.2f" % (delta * 2))
                            + "$"
                        )
            if jobItem.result_converge is not None and jobItem.result_converge.R_eigs:
                caption.append("$R-1 =" + jobItem.result_converge.worstR() + "$")
            if jobItem.result_marge is not None:
                if args.systematic_average:
                    table_lines += get_systematic_average_table_lines(
                        jobItem, _referenceDataJobItem
                    )
                else:
                    table_lines += get_table_lines(
                        jobItem.result_marge, _referenceDataJobItem
                    )

        table_lines.append("")
        if not args.forpaper:
            table_lines.append("; ".join(caption))
        if bf is not None and not args.forpaper:
            table_lines.append("")
            table_lines.append("$\\chi^2_{\\rm eff}$:")
            if deltaChisqJobItem is not None:
                compChiSq = deltaChisqJobItem.result_bestfit
            else:
                compChiSq = None
            for kind, vals in bf.sortedChiSquareds():
                table_lines.append(kind + " - ")
                for val in vals:
                    line = (
                        "  "
                        + tex_escape_text(val.name)
                        + ": "
                        + ("%.2f" % val.chisq)
                        + " "
                    )
                    if compChiSq is not None:
                        comp = compChiSq.chiSquareForKindName(kind, val.name)
                        if comp is not None:
                            line += r"($\Delta$ " + ("%.2f" % (val.chisq - comp)) + ") "
                    table_lines.append(line)
        return table_lines

    def compare_table(jobItems, titles=None):
        for job_i in jobItems:
            job_i.loadJobItemResults(
                paramNameFile=args.paramNameFile,
                bestfit=not args.nobestfit,
                bestfitonly=args.bestfitonly,
            )
            print(job_i.name)
        if titles is None:
            titles = [
                job_i.datatag for job_i in jobItems if job_i.result_marge is not None
            ]
        else:
            titles = titles.split(";")
        return types.ResultTable(
            1,
            [job_i.result_marge for job_i in jobItems if job_i.result_marge is not None],
            limit=args.limit,
            titles=titles,
            blockEndParams=args.blockEndParams,
            paramList=args.paramList,
        ).lines

    if args.changes_replacing is not None:
        if args.data is not None:
            args.data += args.changes_replacing

    items = opts.sortedParamtagDict(chainExist=not args.bestfitonly)

    if args.all_limits:
        limits = [1, 2, 3]
    else:
        limits = [args.limit]

    if args.changes_from_paramtag is not None:
        if args.changes_from_datatag is not None:
            raise Exception(
                "You cannot have both changes_from_paramtag and changes_from_datatag"
            )
        if (
            args.delta_chisq_paramtag is not None
            and args.delta_chisq_paramtag != args.changes_from_paramtag
        ):
            raise Exception(
                "when using changes_from_paramtag, "
                "delta_chisq_paramtag is set equal to that"
            )
        if args.no_delta_chisq:
            raise Exception("when using changes_from_paramtag cannot have no_delta_chisq")
        args.delta_chisq_paramtag = args.changes_from_paramtag

    def data_index(job_i):
        if args.changes_data_ignore:
            ignores = {}
            for ig in args.changes_data_ignore:
                ignores[ig] = ""
            return job_i.data_set.makeNormedDatatag(ignores)
        else:
            return job_i.normed_data

    baseJobItems = {}
    for paramtag, parambatch in items:
        isBase = len(parambatch[0].param_set) == 0
        for job_item in parambatch:
            if (
                args.delta_chisq_paramtag is None
                and isBase
                and not args.no_delta_chisq
                or args.delta_chisq_paramtag is not None
                and job_item.paramtag == args.delta_chisq_paramtag
            ):
                referenceJobItem = copy.deepcopy(job_item)
                referenceJobItem.loadJobItemResults(paramNameFile=args.paramNameFile)
                baseJobItems[job_item.normed_data] = referenceJobItem

    loc = os.path.split(args.latex_filename)[0]
    if loc:
        os.makedirs(loc, exist_ok=True)

    for limit in limits:
        args.limit = limit

        outfile = args.latex_filename
        if args.all_limits:
            outfile += "_limit" + str(limit)
        if outfile[-4:] != ".tex":
            outfile += ".tex"

        lines = []
        if not args.forpaper:
            lines.append("\\documentclass[10pt]{article}")
            lines.append("\\usepackage[pdftex]{hyperref}")
            lines.append(
                "\\usepackage[paperheight="
                + args.height
                + ",paperwidth="
                + args.width
                + ",margin=0.8in]{geometry}"
            )
            lines.append("\\renewcommand{\\arraystretch}{1.5}")
            lines.append("\\begin{document}")
            if args.header_tex is not None:
                lines.append(open(args.header_tex).read())
            lines.append("\\tableofcontents")

        # set of baseline results, e.g. for Delta chi^2

        for paramtag, parambatch in items:
            isBase = len(parambatch[0].param_set) == 0
            if not args.forpaper:
                if isBase:
                    paramText = "Baseline model"
                else:
                    paramText = tex_escape_text("+".join(parambatch[0].param_set))
                section = "\\newpage\\section{ " + paramText + "}"
            else:
                section = ""
            if args.compare is not None:
                compares = opts.filterForDataCompare(parambatch, args.compare)
                if len(compares) == len(args.compare):
                    lines.append(section)
                    lines += compare_table(compares, args.titles)
                else:
                    print("no matches for compare: " + paramtag)
            else:
                lines.append(section)
                theseItems = [
                    jobItem
                    for jobItem in parambatch
                    if (os.path.exists(jobItem.distPath) or args.bestfitonly)
                    and (
                        args.converge == 0 or jobItem.hasConvergeBetterThan(args.converge)
                    )
                ]

                referenceDataJobItem = None
                if args.changes_from_datatag is not None:
                    for job_item in theseItems:
                        if (
                            job_item.normed_data == args.changes_from_datatag
                            or job_item.datatag == args.changes_from_datatag
                        ):
                            referenceDataJobItem = copy.deepcopy(job_item)
                            referenceDataJobItem.loadJobItemResults(
                                paramNameFile=args.paramNameFile, bestfit=args.bestfitonly
                            )
                if args.changes_adding_data is not None:
                    baseJobItems = {}
                    refItems = []
                    for job_item in theseItems:
                        if job_item.data_set.hasName(args.changes_adding_data):
                            job_item.normed_without = "_".join(
                                sorted(
                                    [
                                        x
                                        for x in job_item.data_set.names
                                        if x not in args.changes_adding_data
                                    ]
                                )
                            )
                            refItems.append(job_item.normed_without)
                        else:
                            job_item.normed_without = None
                    for job_item in theseItems:
                        if job_item.normed_data in refItems:
                            referenceJobItem = copy.deepcopy(job_item)
                            referenceJobItem.loadJobItemResults(
                                paramNameFile=args.paramNameFile, bestfit=args.bestfitonly
                            )
                            baseJobItems[job_item.normed_data] = referenceJobItem
                if args.changes_replacing is not None:
                    origCompare = [
                        item
                        for item in theseItems
                        if args.changes_replacing[0] in item.data_set.names
                    ]
                    baseJobItems = {}
                    for job_item in origCompare:
                        referenceJobItem = copy.deepcopy(job_item)
                        referenceJobItem.loadJobItemResults(
                            paramNameFile=args.paramNameFile, bestfit=args.bestfitonly
                        )
                        baseJobItems[job_item.normed_data] = referenceJobItem

                for job_item in theseItems:
                    if args.changes_adding_data is not None:
                        if job_item.normed_without is not None:
                            referenceDataJobItem = baseJobItems.get(
                                job_item.normed_without, None
                            )
                        else:
                            referenceDataJobItem = None
                        referenceJobItem = referenceDataJobItem
                        if args.changes_only and not referenceDataJobItem:
                            continue
                    elif args.changes_replacing is not None:
                        referenceDataJobItem = None
                        for replace in args.changes_replacing[1:]:
                            if replace in job_item.data_set.names:
                                referenceDataJobItem = baseJobItems.get(
                                    batch.normalizeDataTag(
                                        job_item.data_set.tagReplacing(
                                            replace, args.changes_replacing[0]
                                        )
                                    ),
                                    None,
                                )
                                break
                        referenceJobItem = referenceDataJobItem
                        if args.changes_only and not referenceDataJobItem:
                            continue
                    else:
                        referenceJobItem = baseJobItems.get(data_index(job_item), None)
                    if args.changes_from_paramtag is not None:
                        referenceDataJobItem = referenceJobItem
                    if args.systematic_average and referenceDataJobItem is None:
                        continue
                    if not args.forpaper:
                        if args.systematic_average:
                            lines.append(
                                "\\subsection{ "
                                + tex_escape_text(job_item.name)
                                + "/"
                                + tex_escape_text(referenceDataJobItem.name)
                                + "}"
                            )
                        else:
                            lines.append(
                                "\\subsection{ " + tex_escape_text(job_item.name) + "}"
                            )
                    try:
                        tableLines = param_result_table(
                            job_item, referenceJobItem, referenceDataJobItem
                        )
                        if args.separate_tex:
                            types.TextFile(tableLines).write(job_item.distRoot + ".tex")
                        lines += tableLines
                    except Exception as e:
                        print("ERROR: " + job_item.name)
                        print("Index Error:" + str(e))

        if not args.forpaper:
            lines.append("\\end{document}")

        (outdir, outname) = os.path.split(outfile)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        types.TextFile(lines).write(outfile)
        root = os.path.splitext(outname)[0]

        if not args.forpaper:
            print("Now converting to PDF...")
            delext = ["aux", "log", "out", "toc"]
            with working_directory(outdir):
                for _ in range(3):
                    # iterate three times to get table of contents page numbers right
                    os.system("pdflatex " + outname)
                for ext in delext:
                    if os.path.exists(root + "." + ext):
                        os.remove(root + "." + ext)


def grid_param_compare(args=None):
    opts = BatchArgs(
        "Compare parameter constraints over set of models, producing"
        "latext column content for tables (but not full tables)"
    )
    opts.parser.add_argument(
        "--params", nargs="+", help="list of parameters to compare in the output tables"
    )
    opts.parser.add_argument(
        "--single_extparams",
        nargs="+",
        help="list of single parameter extensions to restrict to",
    )

    opts.parser.add_argument(
        "--compare", nargs="+", default=None, help="list of data tags to compare"
    )
    opts.parser.add_argument("--nobestfits", action="store_true")
    opts.parser.add_argument("--limit", type=int, default=2)
    opts.parser.add_argument(
        "--latex_filename",
        default=None,
        help="name of latex file to produce (otherwise stdout)",
    )
    opts.parser.add_argument(
        "--math_columns",
        action="store_true",
        help="whether to include $$ in the output for each column",
    )
    opts.parser.add_argument("--endline", default="\\cr")
    opts.parser.add_argument(
        "--paramNameFile",
        help="Optional parameter name file giving latex labels for each parameter",
    )

    (batch, args) = opts.parseForBatch(args)
    formatter = types.TableFormatter()

    if args.paramNameFile:
        names = paramnames.ParamNames(args.paramNameFile)
        par_labels = {par: names.parWithName(par, True).label for par in names.names()}
    else:
        par_labels = {}

    if args.compare:
        args.compare = [batch.normalizeDataTag(dat) for dat in args.compare]

    table = {}
    paramtag_for_param = {}
    for par in args.params:
        paramtag_for_param[par] = []

    for jobItem in opts.filteredBatchItems():
        if (args.compare is None or jobItem.matchesDatatag(args.compare)) and (
            not args.single_extparams
            or len(jobItem.param_set) == 1
            and jobItem.hasParam(args.single_extparams)
        ):
            jobItem.loadJobItemResults(
                paramNameFile=None,
                bestfit=not args.nobestfits,
                noconverge=True,
                silent=True,
            )
            if jobItem.result_marge is not None:
                for par in args.params:
                    texValues = jobItem.result_marge.texValues(
                        formatter, par, limit=args.limit
                    )
                    if texValues is not None:
                        if par not in par_labels:
                            par_labels[par] = jobItem.result_marge.parWithName(
                                par, True
                            ).label
                        if jobItem.paramtag not in table:
                            table[jobItem.paramtag] = {}
                        data_table = table[jobItem.paramtag]
                        if jobItem.paramtag not in paramtag_for_param[par]:
                            paramtag_for_param[par].append(jobItem.paramtag)
                        if jobItem.normed_data not in data_table:
                            data_table[jobItem.normed_data] = {}
                        data_table[jobItem.normed_data][par] = texValues

    def make_math(txt):
        if args.math_columns:
            return "$" + txt + "$"
        else:
            return txt

    def text_as_column(txt, latex=True):
        if latex:
            return make_math(txt)
        else:
            return txt

    def sort_data(_batch, datatags):
        items = []
        for tag in datatags:
            items += [item for item in _batch if item == tag]
        return items

    lines = []

    for i, par in enumerate(args.params):
        for paramtag in paramtag_for_param[par]:
            dataTable = table[paramtag]
            print(paramtag, len(dataTable))
            if args.compare is not None:
                dataTable = sort_data(dataTable, args.compare)
            cols = [make_math(par_labels[par])]
            for datatag in dataTable:
                if args.latex_filename is not None:
                    for par2 in args.params:
                        if par2 in table[paramtag][datatag]:
                            res = table[paramtag][datatag][par2]
                            if len(res) > 1:
                                cols.append(text_as_column(res[1]))
                            cols.append(text_as_column(res[0], latex=res[0] != "---"))
                else:
                    print(" ", table[paramtag][datatag], datatag)
            lines.append(" & ".join(cols) + args.endline)

    if args.latex_filename is not None:
        if not args.latex_filename.endswith(".tex"):
            args.latex_filename += ".tex"
        (outdir, outname) = os.path.split(args.latex_filename)
        os.makedirs(os.path.dirname(outdir), exist_ok=True)
        types.TextFile(lines).write(args.latex_filename)


if __name__ == "__main__":
    grid_tables()
