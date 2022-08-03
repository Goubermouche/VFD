require("vstudio")

-- CUDA 
premake.api.register {
    name = "cudaFastMath",
    scope = "config",
    kind = "boolean"
}

premake.api.register {
    name = "cudaFiles",
    scope = "config",
    kind = "table"
}

local function writeBoolean(property, value)
    if value == true or value == "On" then
        premake.w('\t<' .. property .. '>true</' .. property .. '>')
    elseif value == false or value == "Off" then
        premake.w('\t<' .. property .. '>false</' .. property .. '>')
    end
end

local function writeString(property, value)
    if value ~= nil and value ~= '' then
        premake.w('\t<' .. property .. '>' .. value .. '</' .. property .. '>')
    end
end

local function writeTableAsOneString(property, values)
    if values ~= nil then
        writeString(property, table.concat(values, ' '))
    end
end

local function addCompilerProps(cfg)
    premake.w('<CudaCompile>')

    -- Determine architecture to compile for
    if cfg.architecture == "x86_64" or cfg.architecture == "x64" then
        premake.w('\t<TargetMachinePlatform>64</TargetMachinePlatform>')
    elseif cfg.architecture == "x86" then
        premake.w('\t<TargetMachinePlatform>32</TargetMachinePlatform>')
    else
        error("Unsupported Architecture")
    end

    -- Set XML tags to their requested values 
    premake.w('<CodeGeneration></CodeGeneration>')
    writeBoolean('FastMath', cfg.cudaFastMath)

    premake.w('</CudaCompile>')
end

premake.override(premake.vstudio.vc2010.elements, "itemDefinitionGroup", function(oldfn, cfg)
    local items = oldfn(cfg)
    table.insert(items, addCompilerProps)
    return items
end)

local function inlineFileWrite(value)
    premake.w('\t<CudaCompile ' .. 'Include=' .. string.escapepattern('"') .. path.getabsolute(value) ..
        string.escapepattern('"') .. '/>')
end

local function checkForGlob(value)
    matchingFiles = os.matchfiles(value)
    if matchingFiles ~= null then
        table.foreachi(matchingFiles, inlineFileWrite)
    end
end

local function addCUDAFiles(cfg)
    if cfg.cudaFiles ~= null then
        premake.w('<ItemGroup>')
        table.foreachi(cfg.cudaFiles, checkForGlob)
        premake.w('</ItemGroup>')
    end
end

premake.override(premake.vstudio.vc2010.elements, "project", function(oldfn, cfg)
    local items = oldfn(cfg)
    table.insert(items, addCUDAFiles)
    return items
end)

-- Workspace
workspace "FluidEngine"
    architecture "x64"

    configurations
    {
        "Debug",
        "Release"
    }

    flags
	{
		"MultiProcessorCompile"
	}

    startproject "FluidEngine"

outputdir = "{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
VULKAN_SDK = os.getenv("VULKAN_SDK")

IncludeDir = {}

IncludeDir["GLFW"] = "Engine/ThirdParty/GLFW/include"
IncludeDir["Glad"] = "Engine/ThirdParty/Glad/include"
IncludeDir["ImGui"]= "Engine/ThirdParty/imgui"
IncludeDir["glm"]  = "Engine/ThirdParty/glm"
IncludeDir["entt"] = "Engine/ThirdParty/entt/include"
IncludeDir["cereal"] = "Engine/ThirdParty/cereal"
IncludeDir["tinyobjloader"] = "Engine/ThirdParty/tinyobjloader"
IncludeDir["VulkanSDK"] = "%{VULKAN_SDK}/Include"

-- IncludeDir["Renderer"] = "Engine/ThirdParty/Renderer/Renderer/src"

-- SPIR-V
LibraryDir = {}
Library = {}

LibraryDir["VulkanSDK"] = "%{VULKAN_SDK}/Lib"

-- Debug
Library["ShaderC_Debug"] = "%{LibraryDir.VulkanSDK}/shaderc_sharedd.lib"
Library["SPIRV_Cross_Debug"] = "%{LibraryDir.VulkanSDK}/spirv-cross-cored.lib"
Library["SPIRV_Cross_GLSL_Debug"] = "%{LibraryDir.VulkanSDK}/spirv-cross-glsld.lib"

-- Release
Library["ShaderC_Release"] = "%{LibraryDir.VulkanSDK}/shaderc_shared.lib"
Library["SPIRV_Cross_Release"] = "%{LibraryDir.VulkanSDK}/spirv-cross-core.lib"
Library["SPIRV_Cross_GLSL_Release"] = "%{LibraryDir.VulkanSDK}/spirv-cross-glsl.lib"

include "Engine/ThirdParty/GLFW"
include "Engine/ThirdParty/Glad"
include "Engine/ThirdParty/imgui"
-- include "Engine/ThirdParty/Renderer"

project "Engine"
    location "Engine"
    kind "ConsoleApp"
    language "C++"

    targetdir "bin/%{cfg.buildcfg}"

    pchheader "pch.h"
    pchsource "Engine/Source/pch.cpp"

    buildcustomizations "BuildCustomizations/CUDA 11.7"

    files
    {
        "%{prj.name}/Resources/**.*",
        "%{prj.name}/Source/**.h",
        "%{prj.name}/Source/**.cpp",
        "%{prj.name}/Source/**.txt",
        "%{prj.name}/ThirdParty/glm/**.hpp",
        "%{prj.name}/ThirdParty/glm/**.inl",
        "%{prj.name}/ThirdParty/cereal/**.hpp",
        "%{prj.name}/ThirdParty/tinyobjloader/**.h"
    }

    cudaFiles 
    {
        "**.cu",
        "**.cuh"
    }

    includedirs
    {
        "%{prj.name}/$(ProjectDir)Source",
        "%{IncludeDir.GLFW}",
        "%{IncludeDir.Glad}",
        "%{IncludeDir.ImGui}",
        "%{IncludeDir.glm}",
        "%{IncludeDir.entt}",
        "%{IncludeDir.cereal}",
        "%{IncludeDir.tinyobjloader}",
        "%{IncludeDir.VulkanSDK}",
        -- "%{IncludeDir.Renderer}"
    }

    links
    {
        "GLFW",
        "Glad",
        "ImGui",
        "opengl32.lib",
        "cudart.lib"
    }

    filter "system:windows"
        cppdialect "C++20"
        staticruntime "On"
        systemversion "latest"

        defines
        {
            "GLFW_INCLUDE_NONE"
        }

    filter "configurations:Debug"
        defines 
        {
            "DEBUG" 
        }

        links
        {
            "%{Library.ShaderC_Debug}",
			"%{Library.SPIRV_Cross_Debug}",
			"%{Library.SPIRV_Cross_GLSL_Debug}"
        }

        symbols "On"
        staticruntime "off"
  
     filter "configurations:Release"
        defines 
        {
            "NDEBUG" 
        }

        links
		{
			"%{Library.ShaderC_Release}",
			"%{Library.SPIRV_Cross_Release}",
			"%{Library.SPIRV_Cross_GLSL_Release}"
		}
        
        optimize "Full"
        staticruntime "off"
        cudaFastMath "On"